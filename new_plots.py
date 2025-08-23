# replot_here.py
import os, json
import matplotlib
matplotlib.use("Agg")  # safe for headless servers

# If your file isn't named final.py, change this import accordingly:
from final import save_plots

SKIP = {"accuracy", "macro avg", "weighted avg"}

def infer_class_names(history, test_results):
    # Prefer class names from test_results classification report
    if test_results and "classification_report" in test_results:
        keys = [k for k in test_results["classification_report"].keys() if k not in SKIP]
        if keys:
            return keys
    # Fallback: last available per-class metrics in history
    for row in reversed(history):
        cm = row.get("class_metrics", {})
        keys = [k for k in cm.keys() if k not in SKIP]
        if keys:
            return keys
    raise RuntimeError("Could not infer class names from JSONs; provide a classes file or ensure per-class metrics exist.")

def main():
    # Load JSONs from current directory
    with open("training_history.json", "r") as f:
        history = json.load(f)

    test_results = None
    if os.path.exists("test_results.json"):
        with open("test_results.json", "r") as f:
            test_results = json.load(f)

    class_names = infer_class_names(history, test_results)
    print(f"✅ Using {len(class_names)} classes: {class_names}")

    # Save plots under ./plots (save_plots creates subfolders as needed)
    save_plots(history, class_names, ".", test_results)
    print("📊 Plots written to ./plots (and ./plots/confidence_plots if test_results.json exists).")

if __name__ == "__main__":
    main()
