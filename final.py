import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score, average_precision_score, roc_auc_score
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
    parser.add_argument('--data-dir', type=str, default='wcs_cropped_download', help='Path to dataset directory')
    parser.add_argument('--epochs-per-subset', type=int, default=15, help='Epochs per subset')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--min-samples', type=int, default=5000, help='Minimum samples per class')
    parser.add_argument('--num-subsets', type=int, default=20, help='Number of data subsets (groups)')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of data loader workers')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_cv_tos_384', help='Directory to save checkpoints')
    parser.add_argument('--early-stopping', type=int, default=15, help='Patience for early stopping')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--test-split', type=float, default=0.1, help='Fraction of data to use as test set')
    parser.add_argument('--cv-folds', type=int, default=5, help='If >0, run classical stratified k-fold cross-validation with this many folds')
    return parser.parse_args()

def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(i) for i in obj]
    else:
        return obj
    
def validate_subsets(subsets, full_dataset):
    """Ensure all subsets contain all classes"""
    all_classes = set(range(len(full_dataset.classes)))
    for i, subset in enumerate(subsets):
        subset_classes = set([full_dataset.samples[idx][1] for idx in subset.indices])
        if subset_classes != all_classes:
            raise ValueError(f"Subset {i} missing classes! Expected {all_classes}, got {subset_classes}")
    print("✅ All subsets contain all classes")
    
def create_subsets(dataset, n_splits=4):
    """Split data into balanced subsets ensuring all classes are present"""
    if n_splits == 1:
        print("📂 Using full dataset as a single subset.")
        return [Subset(dataset, list(range(len(dataset))))]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = [label for (_, label) in dataset.samples]
    subsets = []

    initial_splits = []
    for _, subset_indices in skf.split(np.zeros(len(labels)), labels):
        subset_classes = set([labels[i] for i in subset_indices])
        if len(subset_classes) != len(dataset.classes):
            print(f"⚠️ Subset missing classes! Expected {len(dataset.classes)}, found {len(subset_classes)}")
            continue
        initial_splits.append(subset_indices)

    combined_indices = []
    for i in range(n_splits):
        combined_indices = list(initial_splits[i]) + combined_indices
        subsets.append(Subset(dataset, combined_indices))

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

def setup_model(num_classes, device, subset_group=1, backbone='resnet50'):
    """Initialize backbone + 2-layer head"""
    if backbone == 'resnet50':
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = base.fc.in_features
        base.fc = nn.Identity()
    elif backbone == 'convnext_tiny':
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_feats = base.classifier[2].in_features
        base.classifier[2] = nn.Identity()
    elif backbone == 'vit_b_16':
        base = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        in_feats = base.heads.head.in_features
        base.heads.head = nn.Identity()
    elif backbone == 'mobilenet_v3_large':
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_feats = base.classifier[3].in_features
        base.classifier[3] = nn.Identity()
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")

    classifier = nn.Sequential(
        nn.Linear(in_feats, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )

    class Net(nn.Module):
        def __init__(self, feature, classifier):
            super().__init__()
            self.feature = feature
            self.classifier = classifier
        def forward(self, x):
            x = self.feature(x)
            # CNNs return (N,C,H,W); ViT returns (N,D)
            if x.ndim == 4:
                x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
            return self.classifier(x)

    model = Net(base, classifier).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Model setup: Group {subset_group}, Backbone={backbone}, Trainable params: {trainable_params:,}")
    return model

def load_checkpoint(model, checkpoint_path, device):
    """Simplified checkpoint loading"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"⚠️ Checkpoint loading failed: {str(e)}")
        return False  

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

def evaluate(model, loader, class_names, criterion, device):
    """Comprehensive evaluation with confidence metrics"""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)

            _, preds = torch.max(probs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # Basic metrics
    val_loss /= len(loader.dataset)
    val_acc = np.mean(all_preds == all_labels)
    top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5)
    
    # Confidence metrics
    confidences = np.max(all_probs, axis=1)
    correct_mask = (all_preds == all_labels)
    
    # 1. Mean Confidence per Species
    mean_conf_correct = []
    mean_conf_incorrect = []
    for class_idx in range(len(class_names)):
        class_mask = (all_preds == class_idx)
        if np.sum(class_mask) > 0:
            mean_conf_correct.append(np.mean(confidences[class_mask & correct_mask]))
            mean_conf_incorrect.append(np.mean(confidences[class_mask & ~correct_mask]))
        else:
            mean_conf_correct.append(np.nan)
            mean_conf_incorrect.append(np.nan)
    
    # 2. Confidence Histograms
    conf_bins = np.linspace(0, 1, 11)  # 0-0.1, 0.1-0.2, etc.
    overall_hist, _ = np.histogram(confidences, bins=conf_bins)
    correct_hist, _ = np.histogram(confidences[correct_mask], bins=conf_bins)
    incorrect_hist, _ = np.histogram(confidences[~correct_mask], bins=conf_bins)
    
    # 3. Calibration Curve
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    for i in range(len(conf_bins)-1):
        in_bin = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if np.sum(in_bin) > 0:
            bin_acc = np.mean(all_preds[in_bin] == all_labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
        else:
            bin_acc = 0.0  # or np.nan
            bin_conf = 0.0  # or np.nan
        bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    
    # Calculate Expected Calibration Error (ECE)
    ece = np.sum(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)) * 
                np.array([np.sum((confidences >= conf_bins[i]) & 
                               (confidences < conf_bins[i+1])) for i in range(len(conf_bins)-1)]) / len(all_preds))
    
    # 4. Top-k Confidence Accuracy
    topk_accuracies = []
    for k in range(1, 6):
        topk_accuracies.append(top_k_accuracy_score(all_labels, all_probs, k=k))
    
    # 5. Rejection Curve
    thresholds = np.linspace(0, 0.95, 20)
    rejection_metrics = []
    for thresh in thresholds:
        keep = confidences >= thresh
        if np.sum(keep) > 0:
            acc = np.mean(all_preds[keep] == all_labels[keep])
            precision = np.sum((all_preds[keep] == all_labels[keep])) / np.sum(keep)
            recall = np.sum((all_preds[keep] == all_labels[keep])) / len(all_preds)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            rejection_rate = 1 - (np.sum(keep) / len(all_preds))
            rejection_metrics.append({
                'threshold': thresh,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'rejection_rate': rejection_rate
            })
    
    # 6. Class-Level Confidence Variance
    class_conf_vars = []
    for class_idx in range(len(class_names)):
        class_mask = (all_preds == class_idx)
        if np.sum(class_mask) > 0:
            class_conf_vars.append(np.std(confidences[class_mask]))
        else:
            class_conf_vars.append(np.nan)
    
    # 7. Confidence Gap
    sorted_probs = np.sort(all_probs, axis=1)
    confidence_gaps = sorted_probs[:, -1] - sorted_probs[:, -2]
    mean_gap_per_class = []
    for class_idx in range(len(class_names)):
        class_mask = (all_preds == class_idx)
        if np.sum(class_mask) > 0:
            mean_gap_per_class.append(np.mean(confidence_gaps[class_mask]))
        else:
            mean_gap_per_class.append(np.nan)
    
    # 8. False High-Confidence Rate
    high_conf_thresh = 0.9
    high_conf_wrong = np.sum((confidences >= high_conf_thresh) & ~correct_mask)
    high_conf_wrong_rate = high_conf_wrong / np.sum(~correct_mask) if np.sum(~correct_mask) > 0 else 0
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'top5_accuracy': top5_acc,
        'mAP': average_precision_score(np.eye(len(class_names))[all_labels], all_probs, average='macro'),
        'AUROC': roc_auc_score(np.eye(len(class_names))[all_labels], all_probs, average='macro', multi_class='ovr'),
        'classification_report': classification_report(all_labels, all_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, normalize='true'),
        'confidence_metrics': {
            'mean_confidence_per_class': {
                'correct': mean_conf_correct,
                'incorrect': mean_conf_incorrect
            },
            'confidence_distribution': {
                'overall': overall_hist.tolist(),
                'correct': correct_hist.tolist(),
                'incorrect': incorrect_hist.tolist(),
                'bins': conf_bins.tolist()
            },
            'calibration': {
                'ece': ece,
                'bin_centers': bin_centers,
                'bin_accuracies': bin_accuracies,
                'bin_confidences': bin_confidences
            },
            'topk_accuracies': topk_accuracies,
            'rejection_curve': rejection_metrics,
            'class_confidence_variability': class_conf_vars,
            'confidence_gap': {
                'overall_mean': np.mean(confidence_gaps),
                'per_class': mean_gap_per_class
            },
            'false_high_confidence_rate': high_conf_wrong_rate
        },
        'predictions': {
            'true_labels': all_labels.tolist(),
            'pred_labels': all_preds.tolist(),
            'probabilities': all_probs.tolist(),
            'confidences': confidences.tolist(),
            'confidence_gaps': confidence_gaps.tolist()
        }
    }

def get_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced classes"""
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

def train_on_subsets(args, backbone_name, full_dataset, device, test_loader=None):
    """Main training loop with 4 subsets (groups)"""
    subsets = create_subsets(full_dataset, args.num_subsets)
    validate_subsets(subsets, full_dataset)
    
    best_acc = 0.0
    history = []
    class_names = full_dataset.classes
    
    for subset_idx in range(args.num_subsets):
        subset_group = subset_idx + 1  # Groups are 1-4
        print(f"\n🌀 Processing subset {subset_idx+1}/{args.num_subsets} (Group {subset_group})")
        print(f"📊 Subset size: {len(subsets[subset_idx])} samples")
        print(f"📊 Class distribution: {get_class_distribution(subsets[subset_idx])}")
        
        # Setup model with correct unfreezing
        model = setup_model(len(class_names), device, subset_group, backbone=backbone_name)
        
        # Load checkpoint if continuing training
        if subset_idx > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{subset_idx-1}.pth')
            if os.path.exists(checkpoint_path):
                load_checkpoint(model, checkpoint_path, device)
        
        # Create train/val split
        train_idx, val_idx = train_test_split(
            subsets[subset_idx].indices,
            test_size=0.2,
            stratify=[full_dataset.samples[idx][1] for idx in subsets[subset_idx].indices],
            random_state=42
        )
        
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(384, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        full_dataset.transform = train_transform
        train_subset = Subset(full_dataset, train_idx)
        sampler = get_weighted_sampler(train_subset)
        
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
            num_workers=min(2, args.num_workers),
            persistent_workers=args.num_workers > 0
        )
        
        # Configure optimizer
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.25, patience=1
        )
        
        # Early stopping tracking
        subset_best_acc = 0.0
        subset_early_stop = 0
        
        for epoch in range(args.epochs_per_subset):
            print(f"\nEpoch {epoch+1}/{args.epochs_per_subset} (Subset {subset_idx+1}, Group {subset_group})")
            start_time = time.time()
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, nn.CrossEntropyLoss(), device
            )
            
            val_results = evaluate(
                model, val_loader, class_names, nn.CrossEntropyLoss(), device
            )
            
            scheduler.step(val_results['accuracy'])
            
            # Save history
            history.append({
                'subset': subset_idx,
                'subset_group': subset_group,
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'top5_acc': val_results['top5_accuracy'],
                'mAP': val_results['mAP'],
                'AUROC': val_results['AUROC'],
                'class_metrics': val_results['classification_report'],
                'lr': optimizer.param_groups[0]['lr'],
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'confusion_matrix': val_results['confusion_matrix'].tolist()
            })
            
            # Early stopping check
            if val_results['accuracy'] > subset_best_acc:
                subset_best_acc = val_results['accuracy']
                subset_early_stop = 0
                best_model_state = model.state_dict()
                
                # Save checkpoint
                checkpoint_path = os.path.join(args.checkpoint_dir, f'best_model_subset_{subset_idx}.pth')
                torch.save({
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'subset': subset_idx,
                    'group': subset_group,
                    'val_acc': subset_best_acc,
                    'class_names': class_names
                }, checkpoint_path)
            else:
                subset_early_stop += 1
                if subset_early_stop >= args.early_stopping:
                    print(f"🛑 Early stopping triggered for subset {subset_idx} after {epoch+1} epochs")
                    break
            
            # Print metrics
            epoch_time = time.time() - start_time
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.2%}")
            print(f"Top-5 Acc: {val_results['top5_accuracy']:.2%} | mAP: {val_results['mAP']:.2%}")
            print(f"Time: {epoch_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Optionally evaluate on test set during training
            if test_loader and epoch % 2 == 0:  # Every 2 epochs
                test_results = evaluate(
                    model, test_loader, class_names, nn.CrossEntropyLoss(), device
                )
                print(f"\nTest Acc: {test_results['accuracy']:.2%} | Top-5: {test_results['top5_accuracy']:.2%}")
    
    return history, model


def train_with_cross_validation(args, backbone_name, full_dataset, device, test_loader=None):
    """
    Stratified K-Fold cross-validation training.
    Fixed to ensure validation data is completely separate from training data.
    """
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    labels = [lbl for _, lbl in full_dataset.samples]
    class_names = full_dataset.classes

    history = []
    best_overall_acc = -1.0
    best_model_state = None
    best_fold_id = None

    # Define transforms once
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n🧩 Fold {fold+1}/{args.cv_folds}")
        print(f"📊 Train size: {len(train_idx)} | Val size: {len(val_idx)}")
        
        # Create SEPARATE dataset copies for train and val
        train_dataset = datasets.ImageFolder(args.data_dir)
        train_dataset.samples = [full_dataset.samples[i] for i in train_idx]
        train_dataset.targets = [s[1] for s in train_dataset.samples]
        train_dataset.classes = class_names
        train_dataset.class_to_idx = full_dataset.class_to_idx
        train_dataset.transform = train_transform
        
        val_dataset = datasets.ImageFolder(args.data_dir)
        val_dataset.samples = [full_dataset.samples[i] for i in val_idx]
        val_dataset.targets = [s[1] for s in val_dataset.samples]
        val_dataset.classes = class_names
        val_dataset.class_to_idx = full_dataset.class_to_idx
        val_dataset.transform = val_transform

        # Fresh model each fold
        model = setup_model(num_classes=len(class_names), device=device, subset_group=1, backbone=backbone_name)

        # Create loaders with separate datasets
        sampler = get_weighted_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(2, args.num_workers),
            persistent_workers=args.num_workers > 0
        )

        # Optimizer / scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.25, patience=1
        )

        # Early stopping per fold
        fold_best_acc = -1.0
        fold_patience = 0
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs_per_subset):
            print(f"\nEpoch {epoch+1}/{args.epochs_per_subset} (Fold {fold+1})")
            start_time = time.time()

            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_results = evaluate(model, val_loader, class_names, criterion, device)
            scheduler.step(val_results['accuracy'])

            # Save history row (keep keys compatible with save_plots)
            history.append({
                'subset': fold,                     # treat "fold" as "subset" for plotting compatibility
                'subset_group': 1,                  # not used here but kept for consistency
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_results['loss'],
                'val_acc': val_results['accuracy'],
                'top5_acc': val_results['top5_accuracy'],
                'mAP': val_results['mAP'],
                'AUROC': val_results['AUROC'],
                'class_metrics': val_results['classification_report'],
                'lr': optimizer.param_groups[0]['lr'],
                'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'confusion_matrix': val_results['confusion_matrix'].tolist()
            })

            # Track fold best / early stopping
            if val_results['accuracy'] > fold_best_acc:
                fold_best_acc = val_results['accuracy']
                fold_patience = 0
                # Save best for this fold
                ckpt_path = os.path.join(args.checkpoint_dir, f'best_model_fold_{fold}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'fold': fold,
                    'val_acc': fold_best_acc,
                    'class_names': class_names
                }, ckpt_path)
            else:
                fold_patience += 1
                if fold_patience >= args.early_stopping:
                    print(f"🛑 Early stopping triggered for fold {fold} after {epoch+1} epochs")
                    break

            # Log epoch summary
            elapsed = time.time() - start_time
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"Val   Loss: {val_results['loss']:.4f} | Acc: {val_results['accuracy']:.2%}")
            print(f"Top-5: {val_results['top5_accuracy']:.2%} | mAP: {val_results['mAP']:.2%} | AUROC: {val_results['AUROC']:.2%}")
            report = val_results['classification_report']
            per_class_acc = [(cls, metrics['recall']) for cls, metrics in report.items() if cls in class_names]
            lowest_5 = sorted(per_class_acc, key=lambda x: x[1])[:5]
            print("\n📈 Worst 5 classes by accuracy (recall):")
            for cls, acc in lowest_5:
                print(f"{cls}: {acc:.2%}")
            print(f"Time: {elapsed:.1f}s | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Track overall best fold
        if fold_best_acc > best_overall_acc:
            best_overall_acc = fold_best_acc
            best_fold_id = fold
            best_model_state = model.state_dict().copy()

    # Load the best fold state into a fresh model to return
    best_model = setup_model(num_classes=len(class_names), device=device, subset_group=1, backbone=backbone_name)
    if best_model_state is not None:
        best_model.load_state_dict(best_model_state, strict=False)
        print(f"\n🏅 Best fold: {best_fold_id+1} with Val Acc = {best_overall_acc:.2%}")
    else:
        print("\n⚠️ No best fold state captured; returning last model.")

    return history, best_model


def save_plots(history, class_names, checkpoint_dir, test_results=None):
    """Save comprehensive training plots and metrics"""
    # Create plot directories
    plot_dir = os.path.join(checkpoint_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    if test_results:
        confidence_plot_dir = os.path.join(plot_dir, 'confidence_plots')
        os.makedirs(confidence_plot_dir, exist_ok=True)
        
        # 1. Mean Confidence per Species
        plt.figure(figsize=(12, 6))
        x = np.arange(len(class_names))
        width = 0.35
        plt.bar(x - width/2, test_results['confidence_metrics']['mean_confidence_per_class']['correct'], 
                width, label='Correct Predictions')
        plt.bar(x + width/2, test_results['confidence_metrics']['mean_confidence_per_class']['incorrect'], 
                width, label='Incorrect Predictions')
        plt.xticks(x, class_names, rotation=90)
        plt.xlabel('Class')
        plt.ylabel('Mean Confidence')
        plt.title('Mean Confidence by Prediction Correctness')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(confidence_plot_dir, 'mean_confidence_per_class.png'))
        plt.close()
        
        # 2. Confidence Histograms
        plt.figure(figsize=(12, 6))
        bins = test_results['confidence_metrics']['confidence_distribution']['bins']

        confidences = np.array(test_results['predictions']['confidences'])
        true = np.array(test_results['predictions']['true_labels'])
        pred = np.array(test_results['predictions']['pred_labels'])
        correct_mask = (true == pred)

        plt.hist(confidences, bins=bins, alpha=0.5, label='All')
        plt.hist(confidences[correct_mask], bins=bins, alpha=0.5, label='Correct')
        plt.hist(confidences[~correct_mask], bins=bins, alpha=0.5, label='Incorrect')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.savefig(os.path.join(confidence_plot_dir, 'confidence_histogram.png'))
        plt.close()
        
        # 3. Calibration Curve
        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        calib = test_results['confidence_metrics']['calibration']
        plt.plot(calib['bin_confidences'], calib['bin_accuracies'], 's-', 
                label=f'Model (ECE={calib["ece"]:.3f})')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Actual Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.savefig(os.path.join(confidence_plot_dir, 'calibration_curve.png'))
        plt.close()
        
        # 4. Top-k Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 6), test_results['confidence_metrics']['topk_accuracies'], 'o-')
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.title('Top-k Accuracy')
        plt.xticks(range(1, 6))
        plt.savefig(os.path.join(confidence_plot_dir, 'topk_accuracy.png'))
        plt.close()
        
        # 5. Rejection Curve
        thresholds = [m['threshold'] for m in test_results['confidence_metrics']['rejection_curve']]
        accuracies = [m['accuracy'] for m in test_results['confidence_metrics']['rejection_curve']]
        rejections = [m['rejection_rate'] for m in test_results['confidence_metrics']['rejection_curve']]
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(thresholds, accuracies, 'o-')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence Threshold')
        
        plt.subplot(1, 2, 2)
        plt.plot(thresholds, rejections, 'o-')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('Rejection Rate')
        plt.title('Rejection Rate vs Confidence Threshold')
        plt.tight_layout()
        plt.savefig(os.path.join(confidence_plot_dir, 'rejection_curve.png'))
        plt.close()
        
        # 6. Class-Level Confidence Variance
        plt.figure(figsize=(12, 6))
        plt.bar(class_names, test_results['confidence_metrics']['class_confidence_variability'])
        plt.xticks(rotation=90)
        plt.ylabel('Standard Deviation of Confidence')
        plt.title('Confidence Variability by Class')
        plt.tight_layout()
        plt.savefig(os.path.join(confidence_plot_dir, 'class_confidence_variability.png'))
        plt.close()
        
        # 7. Confidence Gap
        plt.figure(figsize=(12, 6))
        gaps = np.array(test_results['predictions']['confidence_gaps'])
        preds = np.array(test_results['predictions']['pred_labels'])
        box_data = [gaps[preds == i] for i in range(len(class_names))]
        plt.boxplot(box_data, labels=class_names)
        plt.xticks(rotation=90)
        plt.ylabel('Confidence Gap (Top1 - Top2)')
        plt.title('Confidence Gap Distribution by Class')
        plt.tight_layout()
        plt.savefig(os.path.join(confidence_plot_dir, 'confidence_gap.png'))
        plt.close()

    # === Separate training plots ===
    df = pd.DataFrame(history)

    # 1) Validation Accuracy across epochs (0..1)
    plt.figure(figsize=(9, 6))
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['val_acc'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Across Epochs')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'val_accuracy_across_epochs.png'), bbox_inches='tight')
    plt.close()

    # 2) Validation Loss across epochs (force 0..1)
    plt.figure(figsize=(9, 6))
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['val_loss'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Across Epochs')
    plt.ylim(0, 1)  # per your request (may clip if loss > 1)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'val_loss_across_epochs.png'), bbox_inches='tight')
    plt.close()

    # 3) Learning Rate schedule (own figure)
    plt.figure(figsize=(9, 6))
    for subset in df['subset'].unique():
        subset_data = df[df['subset'] == subset]
        plt.plot(subset_data['epoch'], subset_data['lr'], label=f'Subset {subset+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.title('Learning Rate Schedule')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'learning_rate_schedule.png'), bbox_inches='tight')
    plt.close()

    # 4) F1-score per class (bar plot, 0..1)
    #    Aggregate the final F1 per class across subsets (mean of last epoch per subset)
    all_class_metrics = []
    for _, row in df.iterrows():
        for class_name, metrics in row['class_metrics'].items():
            if class_name in class_names:  # Only include actual classes
                rec = dict(metrics)
                rec['class'] = class_name
                rec['subset'] = row['subset']
                rec['epoch'] = row['epoch']
                all_class_metrics.append(rec)

    class_df = pd.DataFrame(all_class_metrics)
    # Take the last epoch per (class, subset), then average across subsets
    final_per_subset = (class_df
                        .sort_values(['subset', 'class', 'epoch'])
                        .groupby(['class', 'subset'], as_index=False)
                        .tail(1))
    final_f1 = (final_per_subset
                .groupby('class', as_index=False)['f1-score']
                .mean()
                .sort_values('f1-score', ascending=False))

    plt.figure(figsize=(12, 6))
    plt.bar(final_f1['class'], final_f1['f1-score'])
    plt.xticks(rotation=90)
    plt.ylabel('F1-score')
    plt.ylim(0, 1)
    plt.title('Final F1-score per Class (mean across subsets)')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'f1_per_class_bar.png'), bbox_inches='tight')
    plt.close()

    
    # Confusion Matrix from last epoch
    last_confusion = history[-1]['confusion_matrix']
    plt.figure(figsize=(12, 10))
    sns.heatmap(last_confusion, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()

def main():
    # -------------------------
    # 0) Config & device
    # -------------------------
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Backbones to run sequentially (no CLI flags needed)
    BACKBONES_TO_RUN = [
        "resnet50",
        #"convnext_tiny",
        #"vit_b_16",
        #"mobilenet_v3_large",  # optional: uncomment to include
    ]

    # -------------------------
    # 1) Load and filter dataset
    # -------------------------
    print("📦 Loading data...")
    full_dataset = datasets.ImageFolder(args.data_dir)

    # Filter classes with insufficient samples
    from collections import defaultdict
    class_counts = defaultdict(int)
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    valid_classes, excluded_classes = [], []
    for cls, idx in full_dataset.class_to_idx.items():
        if class_counts.get(idx, 0) >= args.min_samples:
            valid_classes.append(cls)
        else:
            excluded_classes.append(cls)

    print(f"\n📊 Original class count: {len(full_dataset.classes)}")
    print(f"🚫 Excluded {len(excluded_classes)} classes with < {args.min_samples} samples:")
    print(excluded_classes)
    print(f"✅ Using {len(valid_classes)} valid classes")

    # Remap samples to only include valid classes
    valid_class_indices = [full_dataset.class_to_idx[cls] for cls in valid_classes]
    filtered_samples = [
        (path, valid_class_indices.index(label))
        for path, label in full_dataset.samples
        if label in valid_class_indices
    ]

    # Build filtered datasets
    filtered_dataset = datasets.ImageFolder(args.data_dir)
    filtered_dataset.samples = filtered_samples
    filtered_dataset.targets = [s[1] for s in filtered_samples]
    filtered_dataset.classes = valid_classes
    filtered_dataset.class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}
    print("📊 Filtered class distribution:", json.dumps(get_class_distribution(filtered_dataset), indent=2))

    # Split train+val vs test (unchanged)
    from sklearn.model_selection import train_test_split
    trainval_idx, test_idx = train_test_split(
        list(range(len(filtered_samples))),
        test_size=args.test_split,
        stratify=[s[1] for s in filtered_samples],
        random_state=42
    )

    # Create final train+val dataset
    trainval_dataset = datasets.ImageFolder(args.data_dir)
    trainval_dataset.samples = [filtered_samples[i] for i in trainval_idx]
    trainval_dataset.targets = [s[1] for s in trainval_dataset.samples]
    trainval_dataset.classes = valid_classes
    trainval_dataset.class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}

    # Create test dataset
    test_dataset = datasets.ImageFolder(args.data_dir)
    test_dataset.samples = [filtered_samples[i] for i in test_idx]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.classes = valid_classes
    test_dataset.class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}

    print(f"\n📊 Final train+val size: {len(trainval_dataset)} samples")
    print(f"🧪 Test set size: {len(test_dataset)} samples")
    print("📊 Test set distribution:", json.dumps(get_class_distribution(test_dataset), indent=2))

    # Test transform/loader (unchanged)
    test_transform = transforms.Compose([
        transforms.Resize(448),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset.transform = test_transform
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(2, args.num_workers)
    )

    # -------------------------
    # 2) Run ALL backbones sequentially
    # -------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    all_runs = []

    for backbone in BACKBONES_TO_RUN:
        print(f"\n================  Running backbone: {backbone}  ================\n")

        # Per-backbone subdir to keep artifacts tidy
        run_dir = os.path.join(args.checkpoint_dir, backbone)
        os.makedirs(run_dir, exist_ok=True)

        # Train on your progressive subsets (uses your existing logic)
        history, trained_model = train_with_cross_validation(
            args=args,
            backbone_name=backbone,            # <-- requires tiny signature change in train_on_subsets
            full_dataset=trainval_dataset,
            device=device,
            test_loader=test_loader
        )

        # Final evaluation on test set (unchanged)
        print("\n🏆 Final evaluation on test set...")
        test_results = evaluate(
            trained_model, test_loader, trainval_dataset.classes, nn.CrossEntropyLoss(), device
        )

        # Save per-backbone artifacts (same as you already do, but under run_dir)
        print("\n💾 Saving results...")
        with open(os.path.join(run_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f)
        with open(os.path.join(run_dir, 'test_results.json'), 'w') as f:
            json.dump(convert(test_results), f, indent=2)

        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'class_names': trainval_dataset.classes,
            'test_metrics': test_results
        }, os.path.join(run_dir, 'final_model.pth'))

        # Plots (your existing function)
        if len(history) > 0:
            save_plots(history, trainval_dataset.classes, run_dir, test_results)
            print(f"📊 Saved training plots to {run_dir}/plots/")

        # Collect summary row
        all_runs.append({
            'backbone': backbone,
            'accuracy': test_results['accuracy'],
            'top5_accuracy': test_results['top5_accuracy'],
            'mAP': test_results['mAP'],
            'AUROC': test_results['AUROC'],
            'ECE': test_results['confidence_metrics']['calibration']['ece'],
        })

        # Free VRAM between runs
        del trained_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------
    # 3) Cross-backbone summary
    # -------------------------
    summary_df = pd.DataFrame(all_runs)
    summary_csv = os.path.join(args.checkpoint_dir, 'summary_by_backbone.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n✅ Wrote backbone summary to {summary_csv}\n")
    print(summary_df)

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    main()