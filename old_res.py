import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Environment optimizations
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128' if torch.cuda.is_available() else ''

def parse_args():
    parser = argparse.ArgumentParser(description='Progressive ResNet50 Wildlife Classification')
    parser.add_argument('--data-dir', type=str, default='snapshot_safari_10k_crops_deduped', help='Path to dataset directory')
    parser.add_argument('--epochs-per-subset', type=int, default=5, help='Epochs per subset')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min-samples', type=int, default=5000, help='Minimum samples per class')
    parser.add_argument('--num-subsets', type=int, default=20, help='Number of data subsets')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--subset-range', type=str, default='0-19', help='Subset range to process (e.g., "5-19" for subsets 6-20)')
    return parser.parse_args()

def create_subsets(dataset, n_splits=20):
    """Split data into balanced subsets preserving class ratios"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = [label for (_, label) in dataset.samples]
    subsets = []
    
    for _, subset_indices in skf.split(dataset.samples, labels):
        subsets.append(Subset(dataset, subset_indices))
    
    return subsets

def get_class_distribution(dataset):
    """Get class distribution as serializable dictionary"""
    unique_classes, counts = torch.unique(
        torch.tensor([s[1] for s in dataset.samples]), 
        return_counts=True
    )
    return {dataset.classes[int(k)]: int(v) for k, v in zip(unique_classes, counts)}

def setup_model(num_classes, device, subset_group=None):
    """Initialize and configure ResNet50 model with progressive unfreezing"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    
    # Enhanced classifier head (always trainable)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    
    # Freeze all layers initially (will be selectively unfrozen)
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True  # Always train classifier
    
    # Progressive unfreezing based on subset group
    if subset_group is not None:
        if subset_group >= 2:  # Subsets 6-10: Unfreeze layer4
            for name, param in model.named_parameters():
                if name.startswith('layer4'):
                    param.requires_grad = True
        
        if subset_group >= 3:  # Subsets 11-15: Unfreeze layer3 and layer4
            for name, param in model.named_parameters():
                if name.startswith('layer3') or name.startswith('layer4'):
                    param.requires_grad = True
        
        if subset_group >= 4:  # Subsets 16-20: Unfreeze all layers
            for param in model.parameters():
                param.requires_grad = True
    
    model = model.to(device)
    if device.type == 'cpu':
        model = model.float()
    return model

def train_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch"""
    model.train()
    total_loss, correct = 0.0, 0
    
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
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
    
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    val_loss /= len(loader.dataset)
    val_acc = report['accuracy']
    class_metrics = {
        cls: {k: v for k, v in report[cls].items() 
              if k in ['precision', 'recall', 'f1-score']}
        for cls in class_names
    }
    
    return val_loss, val_acc, class_metrics

def train_on_subsets(args, model, full_dataset, device):
    """Distributed training across subsets with progressive unfreezing"""
    subsets = create_subsets(full_dataset, args.num_subsets)
    best_acc = 0.0
    history = []
    class_names = full_dataset.classes
    
    # Handle subset range
    if args.subset_range == 'all':
        subset_indices = range(args.num_subsets)
    else:
        start, end = map(int, args.subset_range.split('-'))
        subset_indices = range(start, end+1)
    
    # Load checkpoint if resuming
    resume_subset = None
    if subset_indices[0] > 0:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{subset_indices[0]-1}.pth')
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from subset {subset_indices[0]-1}")
            resume_subset = subset_indices[0]
        except FileNotFoundError:
            print(f"⚠️ No checkpoint found at {checkpoint_path}, starting fresh")
            resume_subset = None
    
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
                        prev_checkpoint = torch.load(prev_checkpoint_path, weights_only=True)
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

            # Training transforms
            full_dataset.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            train_loader = DataLoader(
                Subset(full_dataset, train_idx),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available()
            )
            
            # Validation transforms
            full_dataset.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            val_loader = DataLoader(
                Subset(full_dataset, val_idx),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=min(2, args.num_workers)
            )
            
            # Configure optimizer
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
            
            # Resume optimizer state if continuing same subset group
            if i > subset_indices[0] and subset_group == prev_group:
                try:
                    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i-1}.pth')
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, weights_only=True)
                        if 'optimizer_state_dict' in checkpoint:
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            print("🔄 Resuming optimizer state")
                except Exception as e:
                    print(f"⚠️ Optimizer state not loaded: {str(e)}")

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=2, verbose=True
            )
            
            for epoch in range(args.epochs_per_subset):
                print(f"\nEpoch {epoch+1}/{args.epochs_per_subset} (Subset {i+1}, Group {subset_group})")
                start_time = time.time()
                
                train_loss, train_acc = train_epoch(
                    model, train_loader, optimizer, nn.CrossEntropyLoss(), device
                )
                
                val_loss, val_acc, class_metrics = validate(
                    model, val_loader, class_names, nn.CrossEntropyLoss(), device
                )
                
                scheduler.step(val_acc)
                
                # Save history
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
                    'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
                })
                
                # Enhanced checkpoint saving
                if val_acc > best_acc:
                    best_acc = val_acc
                    checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{i}.pth')
                    temp_path = f"{checkpoint_path}.tmp"
                    
                    try:
                        # Prepare checkpoint data
                        checkpoint = {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'subset': i,
                            'epoch': epoch,
                            'val_acc': val_acc,
                            'class_metrics': class_metrics,
                            'args': vars(args),
                            'group': subset_group
                        }
                        
                        # Save to temporary file first
                        torch.save(checkpoint, temp_path)
                        
                        # Verify temporary file
                        if not os.path.exists(temp_path):
                            raise RuntimeError("Temporary checkpoint not created")
                        
                        # Get checksum of critical parameters
                        original_checksum = hash(tuple(model.state_dict()['fc.0.weight'].cpu().numpy().ravel()))
                        
                        # Atomic rename
                        os.replace(temp_path, checkpoint_path)
                        
                        # Verify final checkpoint
                        verify_checkpoint = torch.load(checkpoint_path, weights_only=True)
                        loaded_checksum = hash(tuple(verify_checkpoint['model_state_dict']['fc.0.weight'].cpu().numpy().ravel()))
                        
                        if original_checksum != loaded_checksum:
                            raise RuntimeError("Checkpoint verification failed - parameter mismatch")
                        
                        print(f"💾 Saved checkpoint for subset {i} (Group {subset_group})")
                        print(f"📁 Location: {checkpoint_path}")
                        
                    except Exception as e:
                        print(f"🔥 ERROR SAVING CHECKPOINT: {str(e)}")
                        
                        # Emergency save
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
    print("📊 Class distribution:", json.dumps(class_dist, indent=2))
    
    # Verify minimum samples
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