# builds the files main.py needs (same pipeline as notebooks/recommendation_engine.ipynb)
# used for deployment so the server can generate its model at build time
import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from scipy.sparse import save_npz

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BACKEND_DIR, "..", "data", "books.csv")

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

books = pd.read_csv(DATA_PATH)

# filling null values based on dtype
books = books.apply(lambda col: col.fillna(0) if col.dtype == "float64" else col.fillna(""))

books["description"] = books["description"].apply(preprocess)
books["title"] = books["title"].str.lower()

# publishing/marketing words ("new york times bestseller", review quotes, awards) appear in
# many descriptions and would otherwise make unrelated books look similar
MARKETING_WORDS = {
    "new", "york", "times", "bestseller", "bestselling", "best", "selling", "author", "authors",
    "acclaimed", "award", "awards", "winning", "winner", "finalist", "prize", "pick", "picks",
    "review", "reviews", "praise", "weekly", "magazine", "journal", "post", "tribune", "kirkus",
    "publishers", "library", "edition", "copies", "sold", "readers", "major", "motion",
    "picture", "starring", "directed", "screenplay", "film", "movie", "usa", "today",
}
tfidf = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS | MARKETING_WORDS))
tfidf_matrix = tfidf.fit_transform(books["description"])

books.to_pickle(os.path.join(BACKEND_DIR, "books.pkl"))
with open(os.path.join(BACKEND_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
save_npz(os.path.join(BACKEND_DIR, "tfidf_matrix.npz"), tfidf_matrix)

print(f"Model files saved to {BACKEND_DIR}")
