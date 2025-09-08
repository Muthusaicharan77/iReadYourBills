import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class BillDataset(Dataset):
    def __init__(self,images_folder, annotations_file, transform=None):
        self.transform = transform
        self.images_folder = images_folder
        self.annotation_file = annotations_file
        self.label2id = {"shop": 1, "item": 2, "date_time": 3, "total": 4}
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
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        box = sample["boxes"]
        label = sample["labels"]
        return image, box, label
        