import re


def preprocess_lyrics(text):
    """
    Preprocess the lyrics by removing special characters and converting to lowercase.

    This function takes a string of text and performs two main operations:
    1. Removes any special characters and numbers, keeping only alphabetic characters and spaces.
    2. Converts all characters to lowercase for consistency.

    @param text: The lyrics text to be preprocessed.
    @type text: str
    @return: The cleaned and normalized text.
    @rtype: str
    """
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    return text


def preprocess_data(df):
    """
    Preprocess a DataFrame by filtering necessary columns and cleaning lyrics.

    This function performs several preprocessing steps on a pandas DataFrame:
    1. Filters the DataFrame to include only 'tag' and 'lyrics' columns.
    2. Removes any rows that have missing values in these columns.
    3. Applies the preprocess_lyrics function to clean the 'lyrics' column.

    @param df: The DataFrame containing song data.
    @type df: pandas.DataFrame
    @return: The preprocessed DataFrame with only required columns and cleaned lyrics.
    @rtype: pandas.DataFrame
    """
    # Filter columns
    filtered_df = df[['tag', 'lyrics']]

    # Remove missing values
    filtered_df = filtered_df.dropna(subset=['tag', 'lyrics'])

    # Clean lyrics
    filtered_df['lyrics'] = filtered_df['lyrics'].apply(preprocess_lyrics)

    return filtered_df
