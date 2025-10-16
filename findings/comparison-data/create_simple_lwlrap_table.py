import pandas as pd
import os

# Change to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read the CSV
df = pd.read_csv('training_metrics_extracted.csv')

# Create a cumulative epoch counter for each model
df_sorted = df.sort_values(['model', 'training_phase', 'epoch'])

# Create a global epoch number for each model
epoch_counter = {}
global_epochs = []

for idx, row in df_sorted.iterrows():
    model = row['model']
    pretrained = row['pretrained']
    model_key = f"{model}_{pretrained}"

    if model_key not in epoch_counter:
        epoch_counter[model_key] = 0
    global_epochs.append(epoch_counter[model_key])
    epoch_counter[model_key] += 1

df_sorted['global_epoch'] = global_epochs

# Filter for epochs 5 and 48
df_filtered = df_sorted[df_sorted['global_epoch'].isin([5, 48])]

# Create a clean model name including pretrained status
def create_model_name(row):
    model = row['model']
    pretrained = row['pretrained']

    model_names = {
        'resnet18': 'ResNet18',
        'mobilenetv4_conv_small': 'MobileNetV4',
        'deit_tiny_patch16_224': 'DeiT (Pretrained)' if pretrained == 'Yes' else 'DeiT (No Pretrain)',
        'convit_tiny': 'ConViT'
    }
    return model_names.get(model, model)

df_filtered['Model'] = df_filtered.apply(create_model_name, axis=1)

# Create the pivot table
result = df_filtered.pivot_table(
    index='Model',
    columns='global_epoch',
    values='lwlrap',
    aggfunc='first'
)

# Rename columns
result.columns = [f'Epoch {int(col)}' for col in result.columns]

# Reset index to make Model a column
result = result.reset_index()

# Round to 4 decimal places
for col in result.columns:
    if col != 'Model':
        result[col] = result[col].round(4)

# Save to CSV
result.to_csv('simple_lwlrap_table.csv', index=False)

# Create markdown table
with open('simple_lwlrap_table.md', 'w') as f:
    f.write("# LWLRAP Scores at Epoch 5 and Epoch 48\n\n")
    f.write(result.to_markdown(index=False, floatfmt='.4f'))
    f.write("\n\n## Notes\n\n")
    f.write("- LWLRAP (Label-Weighted Label-Ranking Average Precision) is the primary evaluation metric\n")
    f.write("- Higher scores are better\n")
    f.write("- Epoch 5: Early training checkpoint\n")
    f.write("- Epoch 48: Mid-to-late training checkpoint\n")

print("✅ Created simple_lwlrap_table.csv")
print("✅ Created simple_lwlrap_table.md")
print("\nTable:")
print(result.to_string(index=False))
