## @package balance_dataset
#  This script balances the dataset of song lyrics by genre.
import pandas as pd

## File path to the CSV file containing the dataset.
#  Replace this with the path to your CSV file.
file_path = './Datasets/subsampled_song_lyrics.csv'

## Read the dataset from the CSV file.
#  The dataset is expected to have at least two columns: 'tag' and 'lyrics'.
#  The 'tag' column represents the genre of each song.
df = pd.read_csv(file_path)

## Find the minimum number of songs in any genre.
#  This is used to balance the dataset so that each genre has the same number of songs.
min_genre_count = df['tag'].value_counts().min()


## Function to sample rows from each genre group.
#  @param x DataFrame grouped by genre.
#  @return A randomly sampled subset of rows from the input DataFrame.
def sample_rows(x):
    return x.sample(min_genre_count)


## Balance the dataset.
#  Applies the sampling function to each genre group.
#  The resulting DataFrame has an equal number of rows for each genre.
balanced_df = df.groupby('tag').apply(sample_rows).reset_index(drop=True)

## Save the balanced dataset to a new CSV file.
#  The balanced dataset is saved in the specified path.
balanced_df.to_csv('./Datasets/balanced_dataset.csv', index=False)

print("Balanced dataset saved as 'balanced_dataset.csv'")
