# Book Recommendation Game 📚  

<img width="500" alt="bookshop_img" src="https://github.com/user-attachments/assets/a0b6b1c4-1764-454f-b895-31c5af96bf5c" />

A bookshop-themed web application that recommends books using content-based filtering.  

Users can enter any book title to get suggestions based purely on content similarity, rather than crowd preferences or historical ratings.  

The recommendation engine is built on a dataset of 7,000 books, using TF-IDF vectorization of book descriptions (after experimenting with Bag of Words) to create a cosine similarity matrix that powers all recommendations.  

When a user searches for a book not in the database, its description is fetched from the Google Books API, vectorized, and compared against the existing matrix, effectively solving the cold start problem.  

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

# Run Notebook to Generate Required Backend Files
jupyter nbconvert --to notebook --execute notebooks/recommendation_engine.ipynb

# Start Backend
cd backend
fastapi dev main.py

# Open Frontend
# Open frontend/index.html in your browser
```

## License   
MIT
