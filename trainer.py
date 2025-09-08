from torchvision import transforms
from dataloader import BillDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
train_dataset = BillDataset(images_folder=r'C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1',annotations_file=r'C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1\annotations.xml', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=lambda x: x)
model = fasterrcnn_resnet50_fpn(pretrained=True)
optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10
model.train()
for epoch in range(epochs):
    
    for batch in train_loader:
        images = []
        targets = []
        for img, boxes, labels in batch:
            images.append(img)
            boxes = torch.tensor(boxes)
            # label_map = {"shop": 1, "item": 2, "date_time": 3, "total": 4}
            # labels = [label_map[l] for l in labels]
            labels = torch.as_tensor(labels)
            # labels = torch.tensor(labels)
            targets.append({"boxes":boxes, "labels":labels})
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
