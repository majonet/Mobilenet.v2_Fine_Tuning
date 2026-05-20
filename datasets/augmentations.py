import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image, ImageEnhance
import numpy as np
import random
import torchvision.transforms.functional as TF


# ------------------------- Helper Functions -------------------------

def photometric_distort(img):
    """Random brightness, contrast, saturation, hue (PhotometricDistort)."""
    img = np.array(img).astype(np.float32)
    # Random brightness
    if random.random() < 0.5:
        delta = random.uniform(-32, 32)
        img += delta

    # Random contrast
    if random.random() < 0.5:
        alpha = random.uniform(0.5, 1.5)
        img *= alpha

    # Convert to HSV for saturation/hue
    img = np.clip(img, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img, mode='RGB')
    pil_img = T.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
    )(pil_img)
    return np.array(pil_img).astype(np.float32)


def expand_image(img, boxes, mean):
    """Randomly place the image on a larger canvas (Expand)."""
    if random.random() < 0.5:
        return img, boxes  # skip expansion

    height, width, depth = img.shape
    scale = random.uniform(1, 2)
    new_h = int(height * scale)
    new_w = int(width * scale)

    left = random.randint(0, new_w - width)
    top = random.randint(0, new_h - height)

    expand_img = np.ones((new_h, new_w, depth), dtype=img.dtype) * mean
    expand_img[top:top + height, left:left + width] = img
    img = expand_img

    boxes = boxes.copy()
    boxes[:, [0, 2]] += left
    boxes[:, [1, 3]] += top

    return img, boxes


def random_sample_crop(img, boxes, labels):
    """Randomly crop image keeping IoU overlap."""
    height, width, _ = img.shape
    if boxes.shape[0] == 0:
        return img, boxes, labels

    for _ in range(50):
        new_w = random.uniform(0.3 * width, width)
        new_h = random.uniform(0.3 * height, height)

        if new_h / new_w < 0.5 or new_h / new_w > 2:
            continue

        left = random.uniform(0, width - new_w)
        top = random.uniform(0, height - new_h)

        rect = np.array([int(left), int(top), int(left + new_w), int(top + new_h)])

        # Compute IoU with existing boxes
        overlap = jaccard_numpy(boxes, rect)
        if overlap.max() < 0.1:
            continue

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        mask = (centers[:, 0] > rect[0]) * (centers[:, 1] > rect[1]) * \
               (centers[:, 0] < rect[2]) * (centers[:, 1] < rect[3])
        if not mask.any():
            continue

        new_boxes = boxes[mask].copy()
        new_labels = labels[mask].copy()

        # Adjust boxes
        new_boxes[:, :2] = np.maximum(new_boxes[:, :2], rect[:2])
        new_boxes[:, 2:] = np.minimum(new_boxes[:, 2:], rect[2:])
        new_boxes[:, :2] -= rect[:2]
        new_boxes[:, 2:] -= rect[:2]

        img = img[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2]), :]
        return img, new_boxes, new_labels

    return img, boxes, labels


def jaccard_numpy(boxes, rect):
    """Compute IoU between each box and a rectangle."""
    inter_xmin = np.maximum(boxes[:, 0], rect[0])
    inter_ymin = np.maximum(boxes[:, 1], rect[1])
    inter_xmax = np.minimum(boxes[:, 2], rect[2])
    inter_ymax = np.minimum(boxes[:, 3], rect[3])
    inter_area = np.maximum(inter_xmax - inter_xmin, 0) * np.maximum(inter_ymax - inter_ymin, 0)

    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
    union_area = box_area + rect_area - inter_area
    return inter_area / np.maximum(union_area, 1e-5)


def random_mirror(img, boxes):
    """Randomly mirror image horizontally."""
    if random.random() < 0.5:
        img = img[:, ::-1, :]
        width = img.shape[1]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return img, boxes


def subtract_means(img, mean):
    return img - mean


# ------------------------- Dataset Class -------------------------

class VOCDataset(Dataset):
    def __init__(self, images, annotations, mean=(123, 117, 104), size=(300, 300)):
        self.images = images
        self.annotations = annotations
        self.mean = np.array(mean, dtype=np.float32)
        self.size = size  # (h,w)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(self.images[idx]).astype(np.float32)
        ann = self.annotations[idx]

        boxes = np.array([a["bbox"] for a in ann], dtype=np.float32) if len(ann) else np.zeros((0, 4))
        labels = np.array([a["class_id"] for a in ann], dtype=np.int64) if len(ann) else np.zeros((0,), dtype=np.int64)

        # ----------- Augmentation Pipeline -----------
        img = photometric_distort(img)
        img, boxes = expand_image(img, boxes, self.mean)
        img, boxes, labels = random_sample_crop(img, boxes, labels)
        img, boxes = random_mirror(img, boxes)

        # Resize to final size 300x300
        h, w, _ = img.shape
        img = TF.to_pil_image(img.astype(np.uint8))
        img = TF.resize(img, self.size)
        img = np.array(img).astype(np.float32)

        # Scale boxes
        scale_x = self.size[1] / w
        scale_y = self.size[0] / h
        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        img = subtract_means(img, self.mean)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return img, boxes, labels


# ------------------------- Collate Function -------------------------

def detection_collate(batch):
    images, boxes, labels = [], [], []
    for img, box, label in batch:
        images.append(img)
        boxes.append(box)
        labels.append(label)
    images = torch.stack(images, dim=0)
    return images, boxes, labels
def train_data(x0, x1):
    # Load dataset images and annotations
    image_paths = list(image_dir.glob('*.jpg'))[:]
    images = [load_image(img_path) for img_path in image_paths]
    annotations = [load_labels(annotation_dir / (img_path.stem + '.xml')) for img_path in image_paths]

    # Use the VOCDataset class that already includes all augmentations
    dataset = VOCDataset(
        images,
        annotations,
        mean=(123, 117, 104),
        size=(300, 300)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,           # shuffle for training
        collate_fn=detection_collate
    )
    return train_loader


# Example usage
train_loader = train_data(1, 2)
print("train_loader created successfully with", len(train_loader.dataset), "samples")
