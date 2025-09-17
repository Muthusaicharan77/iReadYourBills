# from torchvision import transforms
# from dataloader import BillDataset
# from torch.utils.data import DataLoader
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# import torch
# transform = transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
# train_dataset = BillDataset(images_folder=r'C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1',annotations_file=r'C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1\annotations.xml', transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,collate_fn=lambda x: x)
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# optimizer= torch.optim.Adam(model.parameters(), lr=1e-4)
# epochs = 10
# model.train()
# prev_loss = 0
# for epoch in range(epochs):
    
#     for batch in train_loader:
#         # prev_loss = 0
#         images = []
#         targets = []
#         for img, boxes, labels in batch:
            
#             images.append(img)
#             boxes = torch.tensor(boxes)
#             # label_map = {"shop": 1, "item": 2, "date_time": 3, "total": 4}
#             # labels = [label_map[l] for l in labels]
#             labels = torch.as_tensor(labels)
#             # labels = torch.tensor(labels)
#             targets.append({"boxes":boxes, "labels":labels})
#         loss_dict = model(images, targets)
#         loss = sum(loss_dict.values())
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
    
#     if loss.item() < prev_loss:
#         print("saving model")
#         torch.save(model.state_dict(),r"C:\Users\muthu\Documents\bill_ocr\model\model.pth")
#     prev_loss = loss.item()
#     print("prev loss",prev_loss)
#     print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#########################################################################################################
from torchvision import transforms
from dataloader import BillDataset
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import os

# --- Config ---
IMG_SIZE = (512, 512)
NUM_CLASSES = 1 + 4  # background + your 4 classes
SAVE_PATH = r"C:\Users\muthu\Documents\bill_ocr\model\model.pth"
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
transform = transforms.ToTensor()  # resize handled in dataset; ToTensor keeps [0,1]
train_dataset = BillDataset(
    images_folder=r"C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1",
    annotations_file=r"C:\Users\muthu\.cache\kagglehub\datasets\trainingdatapro\ocr-receipts-text-detection\versions\1\annotations.xml",
    transform=transform,
    resize=IMG_SIZE,
)

def collate_fn(batch):
    # returns lists: images, boxes_list, labels_list
    images, boxes, labels = zip(*batch)
    return list(images), list(boxes), list(labels)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# --- Model ---
# Use pretrained backbone weights; replace head to match your classes
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- Training ---
best_loss = float("inf")
model.train()

for epoch in range(EPOCHS):
    running = 0.0
    num_batches = 0

    for images, boxes_list, labels_list in train_loader:
        # Build targets and move to device
        imgs = [img.to(DEVICE) for img in images]
        targets = []
        for b, l in zip(boxes_list, labels_list):
            # Skip samples with 0 boxes (FasterRCNN expects at least one)
            if b.numel() == 0:
                continue
            targets.append({
                "boxes": b.to(DEVICE, dtype=torch.float32),
                "labels": l.to(DEVICE, dtype=torch.int64),
            })

        # If all samples in this batch had no valid boxes, skip
        if len(targets) == 0:
            continue

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
        num_batches += 1

    # Compute epoch average loss
    epoch_loss = running / max(1, num_batches)

    # Save best
    if epoch_loss < best_loss:
        os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), SAVE_PATH)
        best_loss = epoch_loss
        print(f"[Epoch {epoch}] Saved best model. Epoch loss: {epoch_loss:.4f}")

    print(f"Epoch {epoch} | avg loss: {epoch_loss:.4f} | best: {best_loss:.4f}")
