import pandas as pd

def load_data():
    movies_df = pd.read_csv('data/imdb.csv')
    return movies_df