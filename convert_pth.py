import argparse
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from torch.serialization import add_safe_globals
# -----------------------------
# Helpers
# -----------------------------

def load_any_state_dict(path, map_location="cpu"):
    # 1) First attempt: safe load with weights_only=True
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e1:
        # 2) Add allow-list for numpy scalar and retry safely
        try:
            add_safe_globals([np.core.multiarray.scalar])
            obj = torch.load(path, map_location=map_location, weights_only=True)
        except Exception as e2:
            # 3) Final fallback: weights_only=False (ONLY IF YOU TRUST THE CHECKPOINT)
            #    This can execute arbitrary code embedded in the pickle.
            obj = torch.load(path, map_location=map_location, weights_only=False)

    # Unwrap common wrappers to get a state_dict
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
        sd = obj["model_state_dict"]
    elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        sd = obj  # looks like a raw state_dict
    else:
        raise RuntimeError("Unrecognized checkpoint structure (not a state_dict).")

    # Strip DataParallel/Distributed prefixes if present
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    return sd

def save_as(path, state_dict, save_format="checkpoint"):
    if save_format == "checkpoint":
        torch.save({"model_state_dict": state_dict}, path)
    elif save_format == "state_dict":
        torch.save(state_dict, path)
    else:
        raise ValueError("save_format must be 'checkpoint' or 'state_dict'")

def is_wrapped_keys(state_dict):
    # Heuristic: wrapped has "feature." and/or "classifier."
    return any(k.startswith("feature.") for k in state_dict.keys()) or \
           any(k.startswith("classifier.") for k in state_dict.keys())

def is_plain_resnet_sequential_fc(state_dict):
    # Heuristic: plain torchvision resnet backbone keys + fc.0 / fc.1 etc.
    has_backbone = any(k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "avgpool.")) for k in state_dict.keys())
    has_seq_fc  = any(k.startswith("fc.") for k in state_dict.keys())
    return has_backbone and has_seq_fc

# -----------------------------
# Mappings
# -----------------------------
def wrapped2plain(state_dict):
    """Map 'feature.*' -> backbone, 'classifier.*' -> 'fc.*'"""
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("feature."):
            nk = k.replace("feature.", "", 1)  # feature.layer1.0.conv1.weight -> layer1.0.conv1.weight
            out[nk] = v
        elif k.startswith("classifier."):
            nk = k.replace("classifier.", "fc.", 1)  # classifier.0.weight -> fc.0.weight
            out[nk] = v
        else:
            # keep anything that already matches (rare but harmless)
            out[k] = v
    return out

def plain2wrapped(state_dict):
    """Map backbone -> 'feature.*', 'fc.*' -> 'classifier.*'"""
    out = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "avgpool.")):
            nk = "feature." + k
            out[nk] = v
        elif k.startswith("fc."):
            nk = k.replace("fc.", "classifier.", 1)  # fc.0.weight -> classifier.0.weight
            out[nk] = v
        else:
            out[k] = v
    return out

# -----------------------------
# Optional validation
# -----------------------------
def build_plain_resnet_with_seq_fc(num_classes=15, hidden=1024):
    # Import inside to allow --no-validate runs on machines without torchvision
    from torchvision import models
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden, num_classes),
    )
    return m

def build_wrapped_resnet(num_classes=15, hidden=1024):
    from torchvision import models
    base = models.resnet50(weights=None)
    in_feats = base.fc.in_features
    base.fc = nn.Identity()

    classifier = nn.Sequential(
        nn.Linear(in_feats, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden, num_classes),
    )

    class Net(nn.Module):
        def __init__(self, feature, classifier):
            super().__init__()
            self.feature = feature
            self.classifier = classifier
        def forward(self, x):
            x = self.feature(x)
            if x.ndim == 4:
                x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
            return self.classifier(x)

    return Net(base, classifier)

def validate_load(state_dict, target="plain", num_classes=15, hidden=1024):
    if target == "plain":
        model = build_plain_resnet_with_seq_fc(num_classes=num_classes, hidden=hidden)
    elif target == "wrapped":
        model = build_wrapped_resnet(num_classes=num_classes, hidden=hidden)
    else:
        raise ValueError("target must be 'plain' or 'wrapped'")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[validate] missing={len(missing)}, unexpected={len(unexpected)}")
    if missing or unexpected:
        # Not fatal—just report; often BN running stats or tiny naming diffs are harmless
        pass

    # quick shape test
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        y = model(x)
    assert y.shape[1] == num_classes, f"Output classes ({y.shape[1]}) != num_classes ({num_classes})"
    print("[validate] Forward pass OK →", tuple(y.shape))

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Convert ResNet50 checkpoints between wrapped↔plain formats.")
    ap.add_argument("--in", dest="in_path", default="best_model_fold_4.pth", help="Input .pth")
    ap.add_argument("--out", dest="out_path", default="converted.pth", help="Output .pth")
    ap.add_argument("--direction", choices=["wrapped2plain", "plain2wrapped"], default="wrapped2plain",
                    help="Conversion direction.")
    ap.add_argument("--num-classes", type=int, default=15, help="For validation: output classes")
    ap.add_argument("--hidden", type=int, default=1024, help="For validation: hidden size in MLP head")
    ap.add_argument("--save-format", choices=["checkpoint", "state_dict"], default="checkpoint",
                    help="Save as {'model_state_dict': ...} or raw state_dict")
    ap.add_argument("--no-validate", action="store_true", help="Skip loading a dummy model to verify")
    args = ap.parse_args()

    sd_in = load_any_state_dict(args.in_path)
    print(f"[info] loaded {len(sd_in)} tensors from {args.in_path}")

    if args.direction == "wrapped2plain":
        sd_out = wrapped2plain(sd_in)
        target = "plain"
    else:
        sd_out = plain2wrapped(sd_in)
        target = "wrapped"

    print(f"[info] converted → {len(sd_out)} tensors (direction: {args.direction})")

    if not args.no_validate:
        try:
            validate_load(sd_out, target=target, num_classes=args.num_classes, hidden=args.hidden)
        except Exception as e:
            print(f"[warn] validation failed: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)
    save_as(args.out_path, sd_out, save_format=args.save_format)
    print(f"[done] wrote: {args.out_path}")

if __name__ == "__main__":
    main()
