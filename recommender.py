import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class MovieRecommender:
    def __init__(self, movies_df):
        self.movies_df = movies_df
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movie_indices = None
        self._prepare()

    def _prepare(self):
        # Combine useful content features
        self.movies_df['combined'] = (
            self.movies_df['Genre'].fillna('') + ' ' +
            self.movies_df['Certificate'].fillna('') + ' ' +
            self.movies_df['Movie_name'].fillna('')
        )

        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['combined'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.movie_indices = pd.Series(self.movies_df.index, index=self.movies_df['Movie_name']).drop_duplicates()

    def get_recommendations(self, title, top_n=10):
        idx = self.movie_indices.get(title)
        if idx is None:
            return pd.DataFrame()
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.movies_df.iloc[movie_indices][['Movie_name', 'Genre', 'Metascore', 'Rating_from_10']]
