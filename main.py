from functions import SimpleUNet, MarsDataset, train_model, evaluate_checkpoints, plot_results, visualize_result
import torch
import torch.utils.data
import os

cfg = {"path_dataset": "./ai4mars-dataset-merged-0.1",
       "checkpoint_idx": None,
       "learning_rate": 0.001,
       "epochs": 10}

# Load dataset
train_dataset = MarsDataset(root_dir=cfg("path_dataset"), split="train", img_size=128)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MarsDataset(root_dir=cfg("path_dataset"), split="test", img_size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Data loaded..")

# Training
train_model(train_loader, cfg)
print("Training done..")

# Evaluation
results = evaluate_checkpoints(test_loader)
plot_results(results)

# Visualization
best_checkpoint_path = "checkpoint_dir/ckp_9.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet(n_channels=1, n_classes=4).to(device)

if os.path.exists(best_checkpoint_path):
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    visualize_result(model, test_dataset, device)