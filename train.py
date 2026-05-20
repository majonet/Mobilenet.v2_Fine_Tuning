
import torch
import torch.optim as optim

from datasets.voc_dataset import VOCDataset
from datasets.augmentations import *
from utils.priors import generate_ssd_priors, specs, image_size
from losses.multibox_loss import MultiboxLoss
from training.trainer import train
from models.model_setup import model

from vision.ssd.config import mobilenetv1_ssd_config

device = "cuda" if torch.cuda.is_available() else "cpu"

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3,
    weight_decay=1e-4
)

config = mobilenetv1_ssd_config

priors = generate_ssd_priors(specs, image_size, clamp=True)

criterion = MultiboxLoss(
    priors,
    iou_threshold=0.7,
    neg_pos_ratio=3,
    center_variance=0.1,
    size_variance=0.2,
    device=device
)

# ======================================================
# Create train_loader and val_loader before training
# ======================================================

# train_loader = ...
# val_loader = ...

train(
    train_loader=train_loader,
    val_loader=val_loader,
    net=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=20
)
