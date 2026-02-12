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
    if model not in epoch_counter:
        epoch_counter[model] = 0
    global_epochs.append(epoch_counter[model])
    epoch_counter[model] += 1

df_sorted['global_epoch'] = global_epochs

# Pivot the data to create a table with models as columns
pivot_df = df_sorted.pivot(index='global_epoch', columns='model', values='valid_loss')

# Rename columns to shorter names
column_names = {
    'resnet18': 'ResNet18',
    'mobilenetv4_conv_small': 'MobileNetV4',
    'deit_tiny_patch16_224': 'DeiT (pretrained)' if df_sorted[df_sorted['model'] == 'deit_tiny_patch16_224']['pretrained'].iloc[0] == 'Yes' else 'DeiT',
    'convit_tiny': 'ConViT'
}

# Split DeiT models by pretrained status
deit_pretrained = df_sorted[(df_sorted['model'] == 'deit_tiny_patch16_224') & (df_sorted['pretrained'] == 'Yes')]
deit_no_pretrained = df_sorted[(df_sorted['model'] == 'deit_tiny_patch16_224') & (df_sorted['pretrained'] == 'No')]

# Create separate pivots for each model
resnet_data = df_sorted[df_sorted['model'] == 'resnet18'][['global_epoch', 'valid_loss']].set_index('global_epoch')
mobilenet_data = df_sorted[df_sorted['model'] == 'mobilenetv4_conv_small'][['global_epoch', 'valid_loss']].set_index('global_epoch')
deit_pretrained_data = deit_pretrained.reset_index(drop=True).reset_index()[['index', 'valid_loss']].set_index('index')
deit_no_pretrained_data = deit_no_pretrained.reset_index(drop=True).reset_index()[['index', 'valid_loss']].set_index('index')
convit_data = df_sorted[df_sorted['model'] == 'convit_tiny'].reset_index(drop=True).reset_index()[['index', 'valid_loss']].set_index('index')

# Combine all models
result = pd.DataFrame({
    'Epoch': range(max(len(resnet_data), len(mobilenet_data), len(deit_pretrained_data), len(deit_no_pretrained_data), len(convit_data))),
})

result['ResNet18'] = resnet_data.reset_index(drop=True)['valid_loss']
result['MobileNetV4'] = mobilenet_data.reset_index(drop=True)['valid_loss']
result['DeiT_Pretrained'] = deit_pretrained_data.reset_index(drop=True)['valid_loss']
result['DeiT_No_Pretrained'] = deit_no_pretrained_data.reset_index(drop=True)['valid_loss']
result['ConViT'] = convit_data.reset_index(drop=True)['valid_loss']

# Save to CSV
result.to_csv('validation_loss_by_epoch.csv', index=False, float_format='%.6f')

# Create markdown table (first 50 epochs + last 10)
with open('validation_loss_table.md', 'w') as f:
    f.write("# Validation Loss by Epoch - All Models\n\n")
    f.write("## First 50 Epochs\n\n")
    f.write(result.head(50).to_markdown(index=False, floatfmt='.6f'))
    f.write("\n\n## Last 10 Epochs\n\n")
    f.write(result.tail(10).to_markdown(index=False, floatfmt='.6f'))
    f.write("\n\n## Full Table Statistics\n\n")
    f.write(f"- Total epochs: {len(result)}\n")
    f.write(f"- ResNet18: {len(resnet_data)} epochs\n")
    f.write(f"- MobileNetV4: {len(mobilenet_data)} epochs\n")
    f.write(f"- DeiT (Pretrained): {len(deit_pretrained_data)} epochs\n")
    f.write(f"- DeiT (No Pretrained): {len(deit_no_pretrained_data)} epochs\n")
    f.write(f"- ConViT: {len(convit_data)} epochs\n")

    # Best validation losses
    f.write("\n## Best Validation Loss per Model\n\n")
    f.write(f"- ResNet18: {result['ResNet18'].min():.6f}\n")
    f.write(f"- MobileNetV4: {result['MobileNetV4'].min():.6f}\n")
    f.write(f"- DeiT (Pretrained): {result['DeiT_Pretrained'].min():.6f}\n")
    f.write(f"- DeiT (No Pretrained): {result['DeiT_No_Pretrained'].min():.6f}\n")
    f.write(f"- ConViT: {result['ConViT'].min():.6f}\n")

print("✅ Created validation_loss_by_epoch.csv")
print("✅ Created validation_loss_table.md")
print(f"\nTable dimensions: {len(result)} rows × {len(result.columns)} columns")
