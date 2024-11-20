import re
import matplotlib.pyplot as plt
from collections import defaultdict

# File path
file_path = '../finetuning/models/kfold_run/logs/kfold_run_training_log_20241119_145213.txt'

# Initialize storage for data
data = defaultdict(lambda: {"loss": [], "eval_loss": []})

# Read and parse the log file
with open(file_path, "r") as file:
    current_fold = None
    for line in file:
        # Detect the start of a new fold
        fold_match = re.search(r"\[Fold (\d+)]", line)
        if fold_match:
            current_fold = int(fold_match.group(1))
        
        # Detect training loss
        loss_match = re.search(r'"loss": ([\d.]+)', line)
        if loss_match and current_fold is not None:
            data[current_fold]["loss"].append(float(loss_match.group(1)))
        
        # Detect evaluation loss
        eval_loss_match = re.search(r'"eval_loss": ([\d.]+)', line)
        if eval_loss_match and current_fold is not None:
            data[current_fold]["eval_loss"].append(float(eval_loss_match.group(1)))

# Plot the results
plt.figure(figsize=(12, 6))

for fold, values in data.items():
    # Plot training loss as a continuous line
    plt.plot(values["loss"], label=f"Fold {fold} - Training Loss")
    
    # Determine evaluation points corresponding to evaluation losses
    eval_x = [i * (len(values["loss"]) // len(values["eval_loss"])) for i in range(len(values["eval_loss"]))]
    
    # Plot evaluation loss as markers
    plt.plot(eval_x, values["eval_loss"], linestyle='-', label=f"Fold {fold} - Eval Loss")

plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss Across Folds")
plt.legend()
plt.grid(True)
plt.show()
