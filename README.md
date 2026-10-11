# Book Recommendation Game 📚  

<img width="500" alt="bookshop_img" src="https://github.com/user-attachments/assets/a0b6b1c4-1764-454f-b895-31c5af96bf5c" />

A bookshop-themed web application that recommends books using content-based filtering.  

Users can enter any book title to get suggestions based purely on content similarity, rather than crowd preferences or historical ratings.  

The recommendation engine is built on a dataset of 7,000 books, using TF-IDF vectorization of book descriptions (after experimenting with Bag of Words) and cosine similarity to find the closest matches. Publishing and marketing phrases like "New York Times bestseller" or award mentions are filtered out, so books are matched on their stories rather than their blurbs. Similarities are computed on demand instead of storing the full similarity matrix, which keeps the server small enough for free hosting.  

When a user searches for a book not in the database (or one without a description), its description is fetched from the Google Books API, vectorized, and compared against the existing TF-IDF vectors, effectively solving the cold start problem.  

## Demo
The game is available online through GitHub Pages:

[Open the Game](https://beyza1tozman.github.io/Book-Recommendation-Game/)

The backend runs on a free server that sleeps when idle, so the first search may take up to a minute.

## Tech Stack   
- **Libraries:** Scikit-learn, Pandas, NumPy, NLTK, SciPy  
- **Backend:** FastAPI  
- **Data:** [Kaggle 7K Books Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) and [Google Books API](https://developers.google.com/books/docs/v1/reference)
- **Frontend:** HTML, CSS, JavaScript, 2D assets created by me

## How to Run the Project Locally

**Prerequisites:** Python 3.12+  

```bash
# Clone the Repository
git clone https://github.com/beyza1tozman/Book-Recommendation-Game.git
cd Book-Recommendation-Game

# Setup Virtual Environment
python -m venv venv        # Windows (Linux/macOS: python3 -m venv venv)
venv\Scripts\activate      # Windows (Linux/macOS: source venv/bin/activate)

# Install Dependencies
pip install -r requirements.txt

# Generate Required Backend Files (same model as the live site)
python backend/build_model.py
# (notebooks/recommendation_engine.ipynb shows the exploration behind it)

# Start Backend
cd backend
fastapi dev main.py

# Open Frontend
# Open frontend/index.html in your browser
```

Optionally, set a `GOOGLE_BOOKS_API_KEY` environment variable before starting the backend. Without one, lookups for books outside the dataset share Google's anonymous quota, which often runs out. If Google Books is unavailable, the backend falls back to the Open Library API.

## Deployment
- **Frontend:** GitHub Pages, deployed from `frontend/` by `.github/workflows/pages.yml` on every push to `main`.
- **Backend:** [Render](https://render.com), configured by `render.yaml`. The model files are generated at build time by `backend/build_model.py`.

## License   
MIT
