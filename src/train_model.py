import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
from collections import defaultdict

# --- Paths based on your folder structure ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # fraud_detection_project/
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")  # processed images
MODELS_DIR = os.path.join(BASE_DIR, "models", "ResNet")  # save trained models in ResNet folder
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Training Parameters ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Reduced for better stability
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

print(f" Using device: {DEVICE}")
print(f" Processed data: {PROCESSED_DIR}")
print(f" Models will be saved to: {MODELS_DIR} (ResNet folder)")


class DocumentDataset(Dataset):
    """Custom dataset for loading processed document images."""

    def __init__(self, processed_dir, transform=None, split='train', train_ratio=0.8):
        self.processed_dir = processed_dir
        self.transform = transform
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        # Auto-detect classes from folder structure
        self._load_dataset(split, train_ratio)

    def _load_dataset(self, split, train_ratio):
        """Load dataset from processed folder structure."""

        # Scan for categories
        categories = [d for d in os.listdir(self.processed_dir)
                      if os.path.isdir(os.path.join(self.processed_dir, d))]

        if not categories:
            raise ValueError(f"No category folders found in {self.processed_dir}")

        categories.sort()  # Ensure consistent ordering
        self.class_names = categories
        self.class_to_idx = {name: idx for idx, name in enumerate(categories)}

        print(f" Found classes: {self.class_names}")
        print(f"  Class mapping: {self.class_to_idx}")

        # Collect all samples
        all_samples = []
        class_counts = defaultdict(int)

        for class_name in categories:
            class_dir = os.path.join(self.processed_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            # Get all image files
            image_files = [f for f in os.listdir(class_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                all_samples.append((img_path, class_idx))
                class_counts[class_name] += 1

        print(f" Dataset distribution:")
        for class_name, count in class_counts.items():
            print(f"   {class_name}: {count} images")

        # Shuffle samples for better distribution
        np.random.seed(42)
        np.random.shuffle(all_samples)

        # Split dataset
        total_samples = len(all_samples)
        train_size = int(total_samples * train_ratio)

        if split == 'train':
            self.samples = all_samples[:train_size]
            print(f" Training set: {len(self.samples)} images")
        else:  # validation
            self.samples = all_samples[train_size:]
            print(f"🔍 Validation set: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            # Load image
            image = Image.open(img_path).convert('RGB')

            # Apply transforms
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', IMG_SIZE, color=(255, 255, 255))
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image, label


def get_transforms():


    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_resnet18_model(num_classes):
    """Create ResNet18 model for document classification."""

    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)

    print(f" Loaded pretrained ResNet18")
    print(f" Original final layer: {model.fc}")

    # Modify final layer for our number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    print(f" Modified final layer: {model.fc}")

    return model


def train_epoch(model, train_loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    train_pbar = tqdm(train_loader, desc="Training", leave=False)

    for batch_idx, (inputs, labels) in enumerate(train_pbar):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

        # Update progress bar
        current_acc = 100.0 * correct_predictions / total_samples
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    val_pbar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Store for detailed metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            current_acc = 100.0 * correct_predictions / total_samples
            val_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = correct_predictions / total_samples

    return epoch_loss, epoch_acc, all_predictions, all_labels


def plot_training_history(history, save_path):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(epochs, [acc * 100 for acc in history['train_acc']], 'b-',
             label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, [acc * 100 for acc in history['val_acc']], 'r-',
             label='Validation Accuracy', linewidth=2)
    ax2.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Confusion matrix saved to: {save_path}")


def save_model_and_results(model, history, class_names, final_metrics):
    """Save trained model and all results."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save model state dict
    model_path = os.path.join(MODELS_DIR, f'resnet18_document_classifier_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)

    # Save complete model info
    model_info = {
        'model_architecture': 'ResNet18',
        'num_classes': len(class_names),
        'class_names': class_names,
        'input_size': IMG_SIZE,
        'timestamp': timestamp,
        'device_used': str(DEVICE)
    }

    # Save training history
    results = {
        'model_info': model_info,
        'training_history': history,
        'final_metrics': final_metrics,
        'training_params': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'num_epochs': NUM_EPOCHS,
            'train_split': TRAIN_SPLIT
        }
    }

    results_path = os.path.join(MODELS_DIR, f'training_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f" Model saved to: {model_path}")
    print(f" Results saved to: {results_path}")

    return model_path, results_path


def main():

    print("FRAUD DETECTION - ResNet18 TRAINING")
    print("=" * 50)


    if not os.path.exists(PROCESSED_DIR):
        print(f" Processed data directory not found: {PROCESSED_DIR}")
        print("Please run data preprocessing first!")
        return


    train_transform, val_transform = get_transforms()


    print("\n Loading datasets...")
    train_dataset = DocumentDataset(PROCESSED_DIR, transform=train_transform,
                                    split='train', train_ratio=TRAIN_SPLIT)
    val_dataset = DocumentDataset(PROCESSED_DIR, transform=val_transform,
                                  split='val', train_ratio=TRAIN_SPLIT)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(" No images found in datasets!")
        return


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)


    print(f"\n Creating ResNet18 model...")
    num_classes = len(train_dataset.class_names)
    model = create_resnet18_model(num_classes).to(DEVICE)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Training setup complete:")
    print(f"   Classes: {train_dataset.class_names}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")


    print(f"\n Starting training for {NUM_EPOCHS} epochs...")
    print("=" * 50)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    best_model_state = None
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\n Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 30)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(model, val_loader, criterion, DEVICE)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"🌟 New best validation accuracy: {val_acc:.4f}")

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Print epoch results
        print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"📊 Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (Val Acc: {best_val_acc:.4f})")

    # Training time
    training_time = time.time() - start_time
    print(f"⏱  Total training time: {training_time / 60:.2f} minutes")

    # Final evaluation
    print(f"\n🔍 Final evaluation...")
    final_val_loss, final_val_acc, final_preds, final_labels = validate_epoch(
        model, val_loader, criterion, DEVICE)

    # Detailed metrics
    report = classification_report(final_labels, final_preds,
                                   target_names=train_dataset.class_names,
                                   output_dict=True)

    final_metrics = {
        'accuracy': final_val_acc,
        'classification_report': report,
        'training_time_minutes': training_time / 60
    }

    # Print results
    print(f"\n🎉 TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print("\nDetailed Classification Report:")
    print(classification_report(final_labels, final_preds,
                                target_names=train_dataset.class_names))

    # Save visualizations
    plot_path = os.path.join(MODELS_DIR, 'training_history.png')
    plot_training_history(history, plot_path)

    cm_path = os.path.join(MODELS_DIR, 'confusion_matrix.png')
    plot_confusion_matrix(final_labels, final_preds, train_dataset.class_names, cm_path)

    # Save model and results
    model_path, results_path = save_model_and_results(model, history,
                                                      train_dataset.class_names,
                                                      final_metrics)

    print(f"\n🎯 Model ready for deployment!")
    print(f"Model file: {model_path}")
    print(f"Results file: {results_path}")


if __name__ == "__main__":
    main()