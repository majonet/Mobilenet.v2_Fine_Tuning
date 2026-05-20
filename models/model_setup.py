import cv2
import torch
import matplotlib.pyplot as plt
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
import copy
model_path_1="/kaggle/input/python-torch-files/model_110_3.pth"
device = torch.device("cuda")
num_classes = 3
model = create_mobilenetv2_ssd_lite(num_classes)
model.load_state_dict(torch.load(model_path_1, map_location=device))
model = model.to(device)
for param in model.parameters():
    param.requires_grad = True

net=model