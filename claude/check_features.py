import pandas as pd
df = pd.read_csv('work/trn_curated_feature.csv', nrows=1)
cols = [c for c in df.columns if c != 'file']
feature_types = set([c.split('_')[0] for c in cols])
print('Feature types found:', sorted(feature_types))
print(f'\nTotal features: {len(cols)}')
