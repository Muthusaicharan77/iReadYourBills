import os
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class BillDataset(Dataset):
    def __init__(self,images_folder, annotations_file, transform=None):
        self.transform = transform
        self.images_folder = images_folder
        self.annotation_file = annotations_file
        tree = ET.parse(annotations_file)
        root = tree.getroot()
        self.samples = []
        for i in root.findall("image"):
            name = i.get("name")
            boxes = []

            for box in root.findall("box"):
                xtl = box.get("xtl")
                ytl = box.get("ytl")
                xbr = box.get("xbr")
                ybr = box.get("ybr")
                boxes.append([xtl,ytl,xbr,ybr])
                label = box.get("label")
            self.samples.append({name:"name",box:"box",label:"label"})
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.images_folder, sample["name"])
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        box = sample["box"]
        label = sample["label"]
        return image, box, label
        