import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from PIL import ImageFile

# Environment optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' if torch.cuda.is_available() else ''
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle corrupted images

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive ResNet50 Wildlife Classification')
    parser.add_argument('--data-dir', type=str, default='snapshot_safari_10k_crops', help='Path to dataset directory')
    parser.add_argument('--epochs-per-subset', type=int, default=5, help='Epochs per subset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--min-samples', type=int, default=4000, help='Minimum samples per class')
    parser.add_argument('--num-subsets', type=int, default=20, help='Number of data subsets')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_384', help='Directory to save checkpoints')
    parser.add_argument('--subset-range', type=str, default='0-19', help='Subset range to process (e.g., "5-19")')
    parser.add_argument('--early-stopping', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Alpha for mixup augmentation')
    return parser.parse_args()

def validate_subsets(subsets, full_dataset):
    """Ensure all subsets contain all classes"""
    all_classes = set(range(len(full_dataset.classes)))
    for i, subset in enumerate(subsets):
        subset_classes = set([full_dataset.samples[idx][1] for idx in subset.indices])
        if subset_classes != all_classes:
            raise ValueError(f"Subset {i} missing classes! Expected {all_classes}, got {subset_classes}")
    print("✅ All subsets contain all classes")

def create_subsets(dataset, n_splits=20):
    """Split data into balanced subsets ensuring all classes are present"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = [label for (_, label) in dataset.samples]
    subsets = []
    
    for _, subset_indices in skf.split(np.zeros(len(labels)), labels):
        subset_classes = set([labels[i] for i in subset_indices])
        if len(subset_classes) != len(dataset.classes):
            print(f"⚠️ Subset missing classes! Expected {len(dataset.classes)}, found {len(subset_classes)}")
            continue
        subsets.append(Subset(dataset, subset_indices))
    
    if len(subsets) < n_splits:
        print(f"⚠️ Only {len(subsets)} valid subsets created (requested {n_splits})")
    return subsets

def get_class_distribution(dataset):
    """Get class distribution as serializable dictionary"""
    if isinstance(dataset, Subset):
        labels = [dataset.dataset.samples[i][1] for i in dataset.indices]
        unique_classes, counts = torch.unique(torch.tensor(labels), return_counts=True)
        return {dataset.dataset.classes[int(k)]: int(v) for k, v in zip(unique_classes, counts)}
    elif hasattr(dataset, 'samples'):
        labels = [s[1] for s in dataset.samples]
        unique_classes, counts = torch.unique(torch.tensor(labels), return_counts=True)
        return {dataset.classes[int(k)]: int(v) for k, v in zip(unique_classes, counts)}
    else:
        raise ValueError("Unsupported dataset type")

def setup_model(num_classes, device, subset_group=None):
    """Initialize and configure ResNet50 model with proper unfreezing"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Classifier head (always trainable)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)  # Fixed to your 17 classes
    )
    
    # Initial setup - only classifier trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():  # This was missing!
        param.requires_grad = True
    
    # Progressive unfreezing
    if subset_group is not None:
        if subset_group >= 2:
            for param in model.layer4.parameters():
                param.requires_grad = True
        if subset_group >= 3:
            for param in model.layer3.parameters():
                param.requires_grad = True
        if subset_group >= 4:
            for param in model.parameters():
                param.requires_grad = True
    
    model = model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model setup: Group {subset_group}, Trainable params: {trainable_params:,}")
    return model

def load_checkpoint(model, checkpoint_path, device):
    """Simplified checkpoint loading"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Load only matching parameters
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"⚠️ Checkpoint loading failed: {str(e)}")
        return False  
    
def mixup_data(x, y, alpha=1.0, device='cuda'):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    """Single training epoch with optional mixup augmentation"""
    model.train()
    total_loss, correct = 0.0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply mixup if enabled
        if mixup_alpha > 0:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, mixup_alpha, device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if mixup_alpha > 0:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
        # For mixup, we calculate accuracy differently
        if mixup_alpha > 0:
            _, preds = torch.max(outputs, 1)
            correct += (lam * preds.eq(targets_a).sum().item() + 
                       (1 - lam) * preds.eq(targets_b).sum().item())
        else:
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def validate(model, loader, class_names, criterion, device):
    """Validation pass with per-class metrics"""
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate comprehensive classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    val_loss /= len(loader.dataset)
    val_acc = report['accuracy']
    class_metrics = {
        cls: {k: v for k, v in report[cls].items() 
              if k in ['precision', 'recall', 'f1-score']}
        for cls in class_names
    }
    
    return val_loss, val_acc, class_metrics, cm_normalized

def get_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced classes"""
    # Handle both Dataset and Subset objects
    if isinstance(dataset, Subset):
        class_counts = get_class_distribution(dataset)
        class_weights = {cls: 1./count for cls, count in class_counts.items()}
        sample_weights = [class_weights[dataset.dataset.classes[label]] 
                         for _, label in [dataset.dataset.samples[i] for i in dataset.indices]]
    else:
        class_counts = get_class_distribution(dataset)
        class_weights = {cls: 1./count for cls, count in class_counts.items()}
        sample_weights = [class_weights[dataset.classes[label]] 
                         for _, label in dataset.samples]
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))
    
def train_on_subsets(args, model, full_dataset, device):
    """Main training loop with all fixes implemented"""
    subsets = create_subsets(full_dataset, args.num_subsets)
    validate_subsets(subsets, full_dataset)
    
    best_acc = 0.0
    history = []
    class_names = full_dataset.classes
    start, end = map(int, args.subset_range.split('-'))
    if start == 0:
        if os.path.exists(args.checkpoint_dir):
            for f in os.listdir(args.checkpoint_dir):
                if f.endswith('.pth'):
                    os.remove(os.path.join(args.checkpoint_dir, f))
    subset_indices = range(start, end+1)
    
    for i in subset_indices:
        print(f"\n🌀 Processing subset {i+1}/{args.num_subsets}")
        print(f"📊 Subset class distribution: {get_class_distribution(subsets[i])}")
        
        # Verify class consistency
        current_classes = set([full_dataset.samples[idx][1] for idx in subsets[i].indices])
        assert len(current_classes) == len(class_names), \
            f"Class count mismatch: {len(current_classes)} vs {len(class_names)}"
        
        # Model setup
        subset_group = (i // 5) + 1
        model = setup_model(len(class_names), device, subset_group)
        
        # Load checkpoint if available
        if i > start:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i-1}.pth')
            if os.path.exists(checkpoint_path):
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    best_acc = checkpoint.get('val_acc', 0.0)  # Initialize from checkpoint
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                except Exception as e:
                    print(f"⚠️ Checkpoint loading failed: {str(e)}")
                    best_acc = 0.0  # Fallback initialization
    
    for i in subset_indices:
        print(f"\n🌀 Processing subset {i+1}/{args.num_subsets}")
        
        # Calculate subset group (1-4)
        subset_group = (i // 5) + 1
        prev_group = ((i-1) // 5) + 1 if i > 0 else 1
        
        # Ensure checkpoint directory exists
        os.makedirs(args.checkpoint_dir, exist_ok=True)

        # Handle group transitions
        if subset_group != prev_group:
            print(f"\n🚀 GROUP TRANSITION DETECTED: Moving from group {prev_group} to {subset_group}")
            model = setup_model(len(class_names), device, subset_group)
            
            # Try to load compatible parameters from previous checkpoint
            prev_checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i-1}.pth')
            if os.path.exists(prev_checkpoint_path):
                try:
                    prev_checkpoint = torch.load(prev_checkpoint_path, map_location=device)
                    current_state = model.state_dict()
                    
                    # Load only matching parameters
                    matched_params = {
                        k: v for k, v in prev_checkpoint['model_state_dict'].items()
                        if k in current_state and v.shape == current_state[k].shape
                    }
                    
                    current_state.update(matched_params)
                    model.load_state_dict(current_state)
                    print(f"🔄 Loaded {len(matched_params)}/{len(current_state)} compatible parameters across group transition")
                except Exception as e:
                    print(f"⚠️ Group transition load failed: {str(e)}")
                    print("🔁 Starting fresh for new group")
            else:
                print("⚠️ No previous checkpoint found for group transition")

        # Create train/val split
        train_idx, val_idx = train_test_split(
            subsets[i].indices,
            test_size=0.2,
            stratify=[full_dataset.samples[idx][1] for idx in subsets[i].indices],
            random_state=42
        )
        
        # Debug check for overlap
        overlap = set(train_idx) & set(val_idx)
        assert len(overlap) == 0, f"Data leakage detected in subset {i}!"

        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Apply transforms to subset
        full_dataset.transform = train_transform
        train_subset = Subset(full_dataset, train_idx)
        
        # Create weighted sampler for imbalanced classes
        sampler = get_weighted_sampler(Subset(full_dataset, train_idx))
        
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0
        )
        
        # Validation transforms
        val_transform = transforms.Compose([
            transforms.Resize(448),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        full_dataset.transform = val_transform
        val_loader = DataLoader(
            Subset(full_dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(1, args.num_workers),
            persistent_workers=args.num_workers > 0
        )
        
        # Configure optimizer with weight decay
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        
        # Resume optimizer state if continuing same subset group
        if i > start and subset_group == prev_group:
            try:
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i-1}.pth')
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'optimizer_state_dict' in checkpoint:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("🔄 Resuming optimizer state")
            except Exception as e:
                print(f"⚠️ Optimizer state not loaded: {str(e)}")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2
        )
        
        # Early stopping tracking per subset
        subset_best_acc = 0.0
        subset_early_stop = 0
        
        for epoch in range(args.epochs_per_subset):
            print(f"\nEpoch {epoch+1}/{args.epochs_per_subset} (Subset {i+1}, Group {subset_group})")
            start_time = time.time()
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, nn.CrossEntropyLoss(), device, args.mixup_alpha
            )
            
            val_loss, val_acc, class_metrics, cm = validate(
                model, val_loader, class_names, nn.CrossEntropyLoss(), device
            )
            
            scheduler.step(val_acc)
            
            # Save history with additional metrics
            history.append({
                'subset': i,
                'subset_group': subset_group,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_metrics': class_metrics,
                'lr': optimizer.param_groups[0]['lr'],
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'confusion_matrix': cm.tolist()  # Save normalized confusion matrix
            })
            
            # Early stopping check
            if val_acc > subset_best_acc:
                subset_best_acc = val_acc
                subset_early_stop = 0
                best_model_state = model.state_dict()
            else:
                subset_early_stop += 1
                if subset_early_stop >= args.early_stopping:
                    print(f"🛑 Early stopping triggered for subset {i} after {epoch+1} epochs")
                    break
            
            # Enhanced checkpoint saving
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i}.pth')
                temp_path = f"{checkpoint_path}.tmp"
                
                try:
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'subset': i,
                        'epoch': epoch,
                        'val_acc': val_acc,
                        'class_metrics': class_metrics,
                        'args': vars(args),
                        'group': subset_group,
                        'class_names': class_names
                    }
                    
                    torch.save(checkpoint, temp_path)
                    os.replace(temp_path, checkpoint_path)
                    print(f"💾 Saved checkpoint for subset {i} (Group {subset_group})")
                    
                except Exception as e:
                    print(f"🔥 ERROR SAVING CHECKPOINT: {str(e)}")
                    emergency_path = f"/tmp/emergency_subset_{i}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'group': subset_group,
                        'epoch': epoch,
                        'val_acc': val_acc
                    }, emergency_path)
                    print(f"🚨 EMERGENCY BACKUP SAVED TO: {emergency_path}")
            
            # Print metrics
            epoch_time = time.time() - start_time
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
            print(f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Trainable params: {history[-1]['trainable_params']:,}")
            
            # Log top/bottom classes
            sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['f1-score'], reverse=True)
            print("\nTop 3 Classes:")
            for cls, metrics in sorted_classes[:3]:
                print(f"{cls}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
            
            print("\nBottom 3 Classes:")
            for cls, metrics in sorted_classes[-3:]:
                print(f"{cls}: Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, F1={metrics['f1-score']:.2f}")
    
    return history

def save_plots(history, class_names, checkpoint_dir):
    """Save comprehensive training plots and metrics"""
    df = pd.DataFrame(history)
    
    # Create plot directory
    plot_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Training and Validation Metrics
    plt.figure(figsize=(15, 10))
    
    # Accuracy plot
    plt.subplot(2, 2, 1)
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['val_acc'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy Across Subsets')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Loss plot
    plt.subplot(2, 2, 2)
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['val_loss'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Loss Across Subsets')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Learning rate plot
    plt.subplot(2, 2, 3)
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['lr'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # F1-scores by class
    plt.subplot(2, 2, 4)
    all_class_metrics = []
    for _, row in df.iterrows():
        for class_name, metrics in row['class_metrics'].items():
            metrics['class'] = class_name
            metrics['subset'] = row['subset']
            metrics['epoch'] = row['epoch']
            all_class_metrics.append(metrics)
    
    class_df = pd.DataFrame(all_class_metrics)
    final_metrics = class_df.groupby(['class', 'subset']).last().reset_index()
    
    sns.boxplot(data=final_metrics, x='class', y='f1-score')
    plt.xticks(rotation=90)
    plt.title('Final F1-scores by Class')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_metrics.png'), bbox_inches='tight')
    plt.close()
    
    # 2. Confusion Matrix from last epoch
    last_confusion = history[-1]['confusion_matrix']
    plt.figure(figsize=(12, 10))
    sns.heatmap(last_confusion, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 3. Class-wise metric trends
    plt.figure(figsize=(15, 8))
    for i, metric in enumerate(['precision', 'recall', 'f1-score']):
        plt.subplot(1, 3, i+1)
        for cls in class_names[:10]:  # Plot first 10 classes for clarity
            cls_data = class_df[class_df['class'] == cls]
            plt.plot(cls_data['epoch'], cls_data[metric], label=cls)
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Trends')
        if i == 2:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_metrics.png'), bbox_inches='tight')
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load dataset
    print("📦 Loading data...")
    full_dataset = datasets.ImageFolder(args.data_dir)
    class_dist = get_class_distribution(full_dataset)
    print("📊 Initial class distribution:", json.dumps(class_dist, indent=2))
    
    # Filter classes with insufficient samples
    valid_classes = []
    excluded_classes = []
    for cls, count in class_dist.items():
        if count >= args.min_samples:
            valid_classes.append(cls)
        else:
            excluded_classes.append(cls)
    
    # Filter the dataset to only include valid classes
    valid_class_indices = [full_dataset.class_to_idx[cls] for cls in valid_classes]
    valid_samples = [
        (path, valid_class_indices.index(label)) 
        for path, label in full_dataset.samples 
        if label in [full_dataset.class_to_idx[cls] for cls in valid_classes]
    ]
    
    # Create new dataset with only valid classes
    filtered_dataset = datasets.ImageFolder(args.data_dir)
    filtered_dataset.samples = valid_samples
    filtered_dataset.targets = [s[1] for s in valid_samples]
    filtered_dataset.classes = valid_classes
    filtered_dataset.class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}
    
    print(f"🚫 Excluded {len(excluded_classes)} classes with < {args.min_samples} samples:")
    print(excluded_classes)
    print("📊 Filtered class distribution:", json.dumps(get_class_distribution(filtered_dataset), indent=2))
    
    # Initialize model with filtered classes
    print("🧠 Initializing model...")
    initial_subset_group = (int(args.subset_range.split('-')[0]) // 5) + 1
    model = setup_model(len(filtered_dataset.classes), device, initial_subset_group)
    
    # Use filtered_dataset instead of full_dataset for training
    history = train_on_subsets(args, model, filtered_dataset, device)
    
    # Save results
    print("\n🏆 Training complete! Saving results...")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    # Generate plots
    if len(history) > 0:
        save_plots(history, full_dataset.classes, args.checkpoint_dir)
        print(f"📊 Saved training plots to {args.checkpoint_dir}/plots/")

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    main()