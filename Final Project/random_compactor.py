import pandas as pd

# Load the dataset
df = pd.read_csv('./Datasets/balanced_dataset.csv')

# Sample 3000, 6000, 12000 songs from each genre
sampled_df_a = df.groupby('tag', group_keys=False).apply(lambda x: x.sample(3000))
sampled_df_b = df.groupby('tag', group_keys=False).apply(lambda x: x.sample(6000))
sampled_df_c = df.groupby('tag', group_keys=False).apply(lambda x: x.sample(12000))

# Save the sampled dataset to a new file
sampled_df_a.to_csv('./Datasets/balanced_dataset_small.csv', index=False)
sampled_df_b.to_csv('./Datasets/balanced_dataset_medium.csv', index=False)
sampled_df_c.to_csv('./Datasets/balanced_dataset_large.csv', index=False)
