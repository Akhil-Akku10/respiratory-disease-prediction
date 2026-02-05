import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.lung_dataset import LungSoundDataset
from models.custom import custommodel
from config import processed_path, batch_size, epoch, learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
dataset = LungSoundDataset(processed_path)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
print("Total samples:", len(dataset))
print("Train samples:", len(train_dataset))
print("Test samples:", len(test_dataset))
counts = torch.tensor([5746, 322, 220, 104, 285], dtype=torch.float)
class_weights = counts.sum() / (len(counts) * counts)
class_weights = class_weights.to(device)
print("Class weights:", class_weights)
model = custommodel(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)
best_bal_acc = 0.0
patience = 7
wait = 0
os.makedirs("checkpoints", exist_ok=True)
for ep in range(epoch):
    model.train()
    running_loss = 0.0
    train_preds = []
    train_labels = []
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(y.cpu().numpy())

    avg_loss = running_loss / len(train_loader)
    bal_acc = balanced_accuracy_score(train_labels, train_preds)
    scheduler.step(avg_loss)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc
        wait = 0
        torch.save(model.state_dict(), "checkpoints/best_model11.pth")
        print("Best model saved!")
    else:
        wait += 1
        print(f"No improvement for {wait} epochs")
        if wait >= patience:
            print("⏹️ Early stopping triggered!")
            break
    print(
        f"Epoch [{ep+1}/{epoch}] "
        f"Loss: {avg_loss:.4f} "
        f"Balanced Acc: {bal_acc:.4f} "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

model.load_state_dict(torch.load("checkpoints/best_model11.pth"))
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)

        preds = torch.argmax(outputs, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(y.cpu().numpy())

print("\ntraining report")
print(classification_report(train_labels, train_preds))


print("\nTest report")
print(classification_report(test_labels, test_preds))


test_bal_acc = balanced_accuracy_score(test_labels, test_preds)

print("Test Balanced Accuracy:", test_bal_acc)
cm = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[0, 1, 2, 3, 4],
    yticklabels=[0, 1, 2, 3, 4]
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("checkpoints/confusion_matrix.png")
plt.show()

torch.save(model.state_dict(), "checkpoints/last_model.pth")
print("Model saved to checkpoints/last_model.pth")
