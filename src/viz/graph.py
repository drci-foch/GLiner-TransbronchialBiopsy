import matplotlib.pyplot as plt
import ast
import json

# Lists to store the data
epochs = []
losses = []
eval_epochs = []
eval_losses = []

# Read training logs
with open('loss.txt', 'r') as f:
    for line in f:
        if line.strip():
            try:
                data = ast.literal_eval(line.strip())
                if 'eval_loss' in data:
                    eval_epochs.append(float(data['epoch']))
                    eval_losses.append(float(data['eval_loss']))
                elif 'loss' in data:
                    epochs.append(float(data['epoch']))
                    losses.append(float(data['loss']))
            except:
                continue

# Get dataset info
with open('src/finetuning/data/data.json', 'r') as f:
    dataset = json.load(f)
    num_examples = len(dataset)
    num_tokens = sum(len(example['tokenized_text']) for example in dataset)
    avg_tokens = num_tokens / num_examples
    labels = set()
    for example in dataset:
        for ner in example['ner']:
            labels.add(ner[2])

# Create figure with two subplots stacked vertically
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[1, 4])

# Add text information in top subplot
info_text = (
    f'Dataset Statistics:\n\n'
    f'Number of examples: {num_examples}\n'
    f'Total tokens: {num_tokens}\n'
    f'Avg tokens/example: {avg_tokens:.1f}\n'
    f'Labels: {", ".join(sorted(labels))}\n'
    f'Final training loss: {losses[-1]:.3f}\n'
    f'Best eval loss: {min(eval_losses):.3f}'
)

ax1.text(0.5, 0.5, info_text,
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax1.transAxes,
         bbox=dict(boxstyle='round',
                  facecolor='white',
                  alpha=0.8),
         fontfamily='monospace',
         fontsize=10)
ax1.axis('off')

# Create loss plot in bottom subplot
ax2.plot(epochs, losses, 'b-', label='Training Loss')
ax2.plot(eval_epochs, eval_losses, 'r.-', label='Evaluation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training and Evaluation Loss Over Time')
ax2.grid(True)
ax2.legend(loc='upper right')
ax2.set_yscale('log')

plt.tight_layout()
plt.show()