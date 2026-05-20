import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import matplotlib.patches as patches

class VOCDataset(Dataset):
    def __init__(self, images, annotations, transform=None, target_size=(300,300)):
        """
        images: لیست یا آرایه numpy images (H,W,C) در فرمت RGB (یا هر فرمتی که تبدیل به PIL شود)
        annotations: لیستِ annotation برای هر تصویر، فرض هر annotation لیستی از dict با {"bbox": [xmin,ymin,xmax,ymax], "class_id": int}
        transform: torchvision.transforms که روی تصویر اعمال می‌شود
        target_size: اندازه‌ای که تصویر به آن Resize می‌شود (width, height)
        """
        self.images = images
        self.annotations = annotations
        self.transform = transform
        self.target_size = target_size  # (H, W) یا (H, W) - توجه: T.Resize از (H,W) می‌پذیرد اما ما از (w,h) استفاده خواهیم کرد

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]   # فرض: numpy array HxWxC (RGB) یا PIL
        ann = self.annotations[idx]

        # تبدیل به PIL اگر لازم باشه (ToPILImage داخل transform هم انجام میده ولی ما برای محاسبه اندازهٔ اصلی نیاز داریم)
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            # اگر tensor بود:
            pil_img = TF.to_pil_image(img)

        orig_w, orig_h = pil_img.size  # PIL: (width, height)
        target_h, target_w = self.target_size  # توجه: قبلاً Resize به (300,300) ست شده است؛ اینجا هم هماهنگ باشه

        # آماده‌سازی bboxes و labels
        if len(ann) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor([obj["bbox"] for obj in ann], dtype=torch.float32)  # [N,4]
            labels = torch.tensor([obj["class_id"] for obj in ann], dtype=torch.long)

            # مقیاس دهی جعبه‌ها: ابتدا محاسبهٔ scale بر اساس اندازهٔ اصلی و هدف
            scale_x = target_w / float(orig_w)
            scale_y = target_h / float(orig_h)

            # فرض فرمت bbox = [xmin, ymin, xmax, ymax]
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x  # x coords
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y  # y coords

        # حالا transform را روی تصویر اعمال کن (که شامل Resize هم هست)
        if self.transform:
            img_transformed = self.transform(np.array(pil_img))
        else:
            img_transformed = TF.to_tensor(pil_img)

        return img_transformed, boxes, labels

def detection_collate(batch):
    images = []
    boxes = []
    labels = []
    for img, box, label in batch:
        images.append(img)
        boxes.append(box)
        labels.append(label)
    images = torch.stack(images, dim=0)
    return images, boxes, labels
def test_data(x0,x1):
    image_paths= list(dataset_dir_im_test.glob('*.jpg'))[:1000] # Use first 5 images
    images = [load_image(img_path) for img_path in image_paths]
    annotations = [load_labels(annotation_dir / (img_path.stem + '.xml')) for img_path in image_paths]
    transform = T.Compose([
        T.ToPILImage(),                 # ورودی: numpy array یا tensor HWC -> PIL
        T.Resize((300,300)),           # MobileNetV2 input size (ولی مقیاس در dataset هم محاسبه می‌شود)
        T.ToTensor()
    ])
    dataset = VOCDataset(images, annotations, transform=transform)
    test_loader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=detection_collate
    )
    return test_loader

test_loader=test_data(1,2)
print("test_loader",test_loader)
