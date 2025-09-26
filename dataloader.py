import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import torch
import cv2
class BillDataset(Dataset):
    def __init__(self,images_folder, annotations_file, transform=None,resize=(512,512)):
        self.transform = transform
        self.images_folder = images_folder
        self.annotation_file = annotations_file
        self.resize = resize
        self.label2id = {"shop": 1, "item": 2, "date_time": 3, "total": 4,"receipts":5}
        tree = ET.parse(annotations_file)
        root = tree.getroot()
        self.samples = []
        for i in root.findall("image"):
            name = i.get("name")
            boxes = []
            labels = []
            for box in i.findall("box"):
                xtl = float(box.get("xtl"))
                ytl = float(box.get("ytl"))
                xbr = float(box.get("xbr"))
                ybr = float(box.get("ybr"))
                boxes.append([xtl,ytl,xbr,ybr])
                label_str  = box.get("label")
                label_id = self.label2id.get(label_str, 0)
                labels.append(label_id)
            self.samples.append({"name": name,"boxes": boxes,"labels": labels})
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.images_folder, sample["name"])
        # image = cv2.imread(image_path)
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        # orig_w, orig_h = image.shape[0],image.shape[1]

        # Resize image
        if self.resize is not None:
            new_w, new_h = self.resize
            sx, sy = new_w / orig_w, new_h / orig_h
            image = image.resize((new_w, new_h))
            # image = cv2.resize(image,(new_w, new_h))
            scaled_boxes = []
            for (xtl, ytl, xbr, ybr) in sample["boxes"]:
                scaled_boxes.append([xtl * sx, ytl * sy, xbr * sx, ybr * sy])
        else:
            scaled_boxes = sample["boxes"]

        # To tensors with correct dtypes
        boxes = torch.tensor(scaled_boxes, dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.int64)

        # Filter out any invalid (label 0) if present
        valid = labels > 0
        boxes = boxes[valid]
        labels = labels[valid]

        if self.transform:
            image = self.transform(image)  # e.g., ToTensor()

        return image, boxes, labels
        