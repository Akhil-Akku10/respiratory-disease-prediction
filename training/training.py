import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, classification_report
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from dataset.lung_dataset import LungSoundDataset
from models.simple_cnn import SimpleCNN
from config import processed_path, batch_size, epoch, learning_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
dataset = LungSoundDataset(processed_path)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

print("Total training samples:", len(dataset))
counts = torch.tensor([5746, 322, 220, 104, 285], dtype=torch.float)

class_weights = counts.sum() / (len(counts) * counts)
class_weights = class_weights.to(device)

print("Class weights:", class_weights)
model = SimpleCNN(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)
best_bal_acc = 0.0

for ep in range(epoch):

    model.train()

    running_loss = 0.0
    all_preds = []
    all_labels = []

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

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())


    avg_loss = running_loss / len(train_loader)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    scheduler.step(avg_loss)
    if bal_acc > best_bal_acc:
        best_bal_acc = bal_acc

        os.makedirs("checkpoints", exist_ok=True)

        torch.save(model.state_dict(), "checkpoints/best_model.pth")

        print("✅ Best model saved!")


    print(
        f"Epoch [{ep+1}/{epoch}] "
        f"Loss: {avg_loss:.4f} "
        f"Balanced Acc: {bal_acc:.4f} "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )
print("\nFinal Classification Report:")
print(classification_report(all_labels, all_preds))
cm = confusion_matrix(all_labels, all_preds)

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
plt.title("Confusion Matrix - Lung Sound Classification")

plt.tight_layout()
plt.savefig("checkpoints/confusion_matrix.png")
plt.show()
torch.save(model.state_dict(), "checkpoints/simple_cnn.pth")
print("Model saved to checkpoints/simple_cnn.pth")