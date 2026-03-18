import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# =========================
# CONFIG
# =========================
TRAIN_DIR = Path(r"E:\Desktop\CLASS\VI\DL\Project\archive\Training")
TEST_DIR = Path(r"E:\Desktop\CLASS\VI\DL\Project\archive\Testing")
MODEL_DIR = Path(r"E:\Desktop\CLASS\VI\DL\Project\models")
MODEL_PATH = MODEL_DIR / "best_brain_tumor_model.pth"
CLASS_NAMES_FILE = Path(r"E:\Desktop\CLASS\VI\DL\Project\class_names.txt")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
NUM_WORKERS = 0  # keep 0 on Windows to avoid issues

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transforms():
    """Return train and test transforms."""
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return train_tfms, test_tfms


def get_dataloaders():
    """Create datasets and dataloaders."""
    train_tfms, test_tfms = get_transforms()

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_tfms)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return train_dataset, test_dataset, train_loader, test_loader


def build_model(num_classes: int):
    """Build ResNet18 model."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def train_one_epoch(model, loader, criterion, optimizer, epoch_num):
    """Train model for one epoch."""
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch_num} [Train]",
        ncols=110
    )

    for inputs, labels in progress_bar:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)

        batch_size_now = labels.size(0)
        running_loss += loss.item() * batch_size_now
        running_correct += (preds == labels).sum().item()
        total_samples += batch_size_now

        current_loss = running_loss / total_samples
        current_acc = running_correct / total_samples

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}"
        )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, epoch_num):
    """Validate model for one epoch."""
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch_num} [Valid]",
        ncols=110
    )

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            batch_size_now = labels.size(0)
            running_loss += loss.item() * batch_size_now
            running_correct += (preds == labels).sum().item()
            total_samples += batch_size_now

            current_loss = running_loss / total_samples
            current_acc = running_correct / total_samples

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}",
                acc=f"{current_acc:.4f}"
            )

    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    return epoch_loss, epoch_acc


def evaluate_model(model, loader, class_names):
    """Print classification report and confusion matrix."""
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    print("\n" + "=" * 60)
    print("FINAL CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("CONFUSION MATRIX")
    print(confusion_matrix(all_labels, all_preds))


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Brain Tumor MRI Training Started")
    print("=" * 60)
    print(f"Device       : {DEVICE}")
    print(f"Train folder : {TRAIN_DIR}")
    print(f"Test folder  : {TEST_DIR}")
    print()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset, test_dataset, train_loader, test_loader = get_dataloaders()
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes      : {class_names}")
    print(f"Train images : {len(train_dataset)}")
    print(f"Test images  : {len(test_dataset)}")
    print()

    with open(CLASS_NAMES_FILE, "w", encoding="utf-8") as file:
        for class_name in class_names:
            file.write(f"{class_name}\n")

    model = build_model(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    total_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n" + "-" * 60)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )

        valid_loss, valid_acc = validate_one_epoch(
            model, test_loader, criterion, epoch
        )

        print(f"\nEpoch {epoch}/{NUM_EPOCHS} Summary")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, MODEL_PATH)
            print(f"Best model saved to: {MODEL_PATH}")

    total_time = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Model saved at         : {MODEL_PATH}")
    print(f"Training time (sec)    : {total_time:.2f}")

    model.load_state_dict(best_weights)
    evaluate_model(model, test_loader, class_names)


if __name__ == "__main__":
    main()