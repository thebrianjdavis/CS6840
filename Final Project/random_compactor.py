import pandas as pd

# Load the dataset
df = pd.read_csv('./Datasets/balanced_dataset.csv')

# Sample 3000 songs from each genre
sampled_df = df.groupby('tag', group_keys=False).apply(lambda x: x.sample(3000))

# Save the sampled dataset to a new file
sampled_df.to_csv('./Datasets/balanced_dataset_small.csv', index=False)
