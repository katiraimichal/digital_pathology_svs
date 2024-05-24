import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from models.resnet50 import CustomResNet50
import torch.optim as optim
from dataset import DPDataset
from trainer import Trainer

torch.cuda.empty_cache()
seed_everything(0, workers=True)

with open('cnfg/hparams.json', 'r') as j:
    config = json.load(j)

# Hyper-parameters
train_name = config["train_name"]
train_csv_dir = config["dirs"]["train_csv_dir"]
val_csv_dir = config["dirs"]["val_csv_dir"]
test_csv_dir = config["dirs"]["test_csv_dir"]
files_path = config["dirs"]["files_dir"]
save_model_path = config["dirs"]["save_model_path"]
checkpoints_path = config["dirs"]["checkpoints_dir"]
learning_rate = config["training"]["learning_rate"]
num_epochs = config["training"]["num_epochs"]
batch_size = config["dataset"]["batch_size"]
preprocess = config["dataset"]["preprocessing"]
transform = config["dataset"]["transform"]

if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)
if not os.path.exists(save_model_path):
    os.makedirs(save_model_path)

# ============
# Data Loading:
# ============
if preprocess is not None:
    svs_preprocess = transforms.Compose([eval(x) for x in preprocess])
if transform is not None:
    svs_transform = transforms.Compose([eval(x) for x in transform])

train_dataset = DPDataset(files_dir=files_path, csv_path=train_csv_dir, train=True, preprocess=svs_preprocess)
val_dataset = DPDataset(files_dir=files_path, csv_path=val_csv_dir, train=False, preprocess=svs_preprocess)
test_dataset = DPDataset(files_dir=files_path, csv_path=test_csv_dir, train=False, preprocess=svs_preprocess)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ======
# Model:
# ======
model = CustomResNet50(num_classes=2, pretrained=True)

criterion = nn.BCELoss(weight=torch.tensor([3.0, 7.0]))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# =========
# Training:
# =========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    save_dir=save_model_path
)

trainer.train()
trainer.test()
