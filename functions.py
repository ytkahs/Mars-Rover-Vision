import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import torchvision.transforms.functional as TF
import time
from sklearn.metrics import jaccard_score
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp

class SimpleUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SimpleUNet, self).__init__()

        # Encoder
        self.conv1 = self.double_conv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self.double_conv(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = self.double_conv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = self.double_conv(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        b = self.bottleneck(p2)

        u2 = self.up2(b)
        u2 = torch.cat((c2, u2), dim=1)
        c_up2 = self.conv_up2(u2)

        u1 = self.up1(c_up2)
        u1 = torch.cat((c1, u1), dim=1)
        c_up1 = self.conv_up1(u1)

        outputs = self.final_conv(c_up1)
        return outputs

class MarsDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        self.images_dir = os.path.join(root_dir, "msl", "images", "edr")
        self.labels_dir = os.path.join(root_dir, "msl", "labels", split)

        if not os.path.exists(self.labels_dir):
            raise FileNotFoundError(f"Error with {self.labels_dir}")

        if split == "test":
            test_subfolder = "masked-gold-min2-100agree"
            self.labels_dir = os.path.join(self.labels_dir, test_subfolder)

            self.label_filenames = [f for f in os.listdir(self.labels_dir) if f.endswith('.png')]
            print(f"Dataset 'test' (subfolder {test_subfolder}) loaded ! {len(self.label_filenames)} images found.")
        else:
            self.label_filenames = [f for f in os.listdir(self.labels_dir) if f.endswith('.png')]
            print(f"Dataset 'train' loaded ! {len(self.label_filenames)} images found.")

    def __len__(self):
        return len(self.label_filenames)

    def __getitem__(self, idx):
        label_name = self.label_filenames[idx]

        if "_merged.png" in label_name:
            image_name = label_name.replace("_merged.png", ".JPG")
        else:
            image_name = label_name.replace(".png", ".JPG")

        label_path = os.path.join(self.labels_dir, label_name)
        image_path = os.path.join(self.images_dir, image_name)

        image = cv2.imread(image_path, 0)
        mask = cv2.imread(label_path, 0)

        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask[mask == 255] = 0
        image = image / 255.0
        image_tensor = torch.from_numpy(image).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).long()

        if self.split == 'train':
          if random.random() > 0.5:
              image_tensor = TF.hflip(image_tensor)
              mask_tensor = TF.hflip(mask_tensor)
          if random.random() > 0.5:
              image_tensor = TF.vflip(image_tensor)
              mask_tensor = TF.vflip(mask_tensor)
        return image_tensor, mask_tensor

def get_model(model_name, n_classes, device):
    if model_name == "simple":
        return SimpleUNet(n_channels=1, n_classes=n_classes).to(device)
    
    elif model_name == "resnet34":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_classes
        )
        return model.to(device)
    else:
        raise ValueError(f"Error with : {model_name}")
    
def train_model(dataloader, cfg):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"processor for training: {device}")

    checkpoint_idx= cfg["checkpoint_idx"]
    model_type = cfg["model_type"]
    num_classes = 4 # 0:soil, 1:bedrock, 2:sand 3:big rock
    lr = cfg["learning_rate"]
    epochs = cfg["epochs"]

    checkpoint_dir = "checkpoint_dir"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = get_model(model_type, num_classes, device)

    start_epoch = 0    
    if checkpoint_idx is not None:
        checkpoint_path = os.path.join(checkpoint_dir, f"ckp_{checkpoint_idx}.pth")
        
        if os.path.exists(checkpoint_path):
            print(f"--> Training will start with : {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            start_epoch = checkpoint_idx 
        else:
            print(f"WARNING : {checkpoint_path} doesn't exist. Training will start at epoch 0")
    
    class_weights = torch.tensor([1.0, 2.0, 1.0, 3.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    model.train()

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        running_loss = 0.0
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step(running_loss/len(dataloader))

        path_name = os.path.join("checkpoint_dir", f"ckp_{epoch+1}.pth")
        torch.save(model.state_dict(), path_name)

        end_time = time.time() - start_time
        minutes = int(end_time // 60)
        secondes = end_time % 60
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(dataloader):.4f} - Time: {minutes} min {secondes:.2f} s")

def evaluate_checkpoints(test_dataloader, cfg, checkpoint_idx=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"processor for evalutation: {device}")

    results = []
    num_classes = 4
    model_type = cfg["model_type"]
    checkpoint_dir = "checkpoint_dir"

    files_to_test = []

    if checkpoint_idx is None:
        files_to_test = os.listdir(checkpoint_dir)
    else:
        filename = f"ckp_{checkpoint_idx}.pth"
        files_to_test = [filename]

    model = get_model(model_type, num_classes, device)
    criterion = nn.CrossEntropyLoss()

    for ckp_name in files_to_test:
        full_path = os.path.join(checkpoint_dir, ckp_name)

        if not os.path.exists(full_path):
            print(f"Error with {full_path}.")
            continue

        model.load_state_dict(torch.load(full_path, map_location=device))
        model.eval()

        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        total_iou = 0.0

        with torch.no_grad():
            for images, masks in test_dataloader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                running_loss += loss.item()

                predicted_classes = torch.argmax(outputs, dim=1)

                correct_pixels += (predicted_classes == masks).sum().item()
                total_pixels += masks.numel()

                preds_flat = predicted_classes.cpu().numpy().flatten()
                masks_flat = masks.cpu().numpy().flatten()
                
                batch_iou = jaccard_score(masks_flat, preds_flat, average='macro', labels=[0, 1, 2, 3], zero_division=0)
                total_iou += batch_iou

        avg_loss = running_loss / len(test_dataloader)
        accuracy = correct_pixels / total_pixels * 100
        mean_iou = (total_iou / len(test_dataloader)) * 100

        epoch_num = int(ckp_name.split('_')[1].split('.')[0])
        results.append((epoch_num, avg_loss, accuracy, mean_iou))

        print(f"[{ckp_name}] Test Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}% | mIoU: {mean_iou:.2f}%")

    return results

def plot_results(results):
    epochs = []
    losses = []
    accuracies = []

    results.sort(key=lambda x: x[0])

    epochs = [x[0] for x in results]
    losses = [x[1] for x in results]
    accuracies = [x[2] for x in results]
    ious = [x[3] for x in results]

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(epochs, losses, 'r-o', label='Test Loss')
    plt.title('Loss evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, accuracies, 'b-o', label='Accuracy')
    plt.title('Accuracy evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, ious, 'g-o', label='IoU')
    plt.title('IoU evolution')
    plt.xlabel('Epoch')
    plt.ylabel('IoU (%)')
    plt.grid(True)
    plt.legend()

    plt.show()


def visualize_result(model, test_dataset, device, image_index=None):
    colors = ['#bdc3c7', '#2980b9', '#f1c40f', '#c0392b'] # Gray, Blue, Yellow, Red
    cmap = ListedColormap(colors)

    class_names = ['Soil', 'Bedrock', 'Sand', 'Big Rocks']

    if image_index is None:
        image_index = np.random.randint(len(test_dataset))
    print(f"image index : {image_index}")

    image_tensor, mask_true_tensor = test_dataset[image_index]

    model.eval()
    with torch.no_grad():
        input_img = image_tensor.unsqueeze(0).to(device)
        output = model(input_img)
        prediction = torch.argmax(output, dim=1)

    img_np = image_tensor.squeeze().numpy()
    mask_true_np = mask_true_tensor.squeeze().numpy()
    mask_pred_np = prediction.cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Real image")
    axes[0].axis('off')

    im1 = axes[1].imshow(mask_true_np, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Groud Truth")
    axes[1].axis('off')

    im2 = axes[2].imshow(mask_pred_np, cmap=cmap, vmin=0, vmax=3)
    axes[2].set_title(f"Prediction")
    axes[2].axis('off')

    legend_patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(4)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.show()