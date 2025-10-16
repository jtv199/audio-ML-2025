# LWLRAP Scores at Epoch 5 and Epoch 48

| Model              |   Epoch 5 |   Epoch 48 |
|:-------------------|----------:|-----------:|
| ConViT             |    0.1218 |     0.2057 |
| DeiT (No Pretrain) |    0.1190 |     0.2060 |
| DeiT (Pretrained)  |    0.1076 |     0.1864 |
| MobileNetV4        |    0.3345 |     0.5752 |
| ResNet18           |    0.3140 |     0.6383 |

## Notes

- LWLRAP (Label-Weighted Label-Ranking Average Precision) is the primary evaluation metric
- Higher scores are better
- Epoch 5: Early training checkpoint
- Epoch 48: Mid-to-late training checkpoint
