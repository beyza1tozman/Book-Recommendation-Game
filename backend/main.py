from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import requests
from urllib.parse import quote
from scipy.sparse import load_npz
from difflib import SequenceMatcher

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# optional, but without a key requests share Google's anonymous quota, which often runs out
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")

# load data
def load_model_data():
    try:
        books = pd.read_pickle(os.path.join(BACKEND_DIR, "books.pkl"))
        with open(os.path.join(BACKEND_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            tfidf = pickle.load(f)
        tfidf_matrix = load_npz(os.path.join(BACKEND_DIR, "tfidf_matrix.npz"))
        return books, tfidf, tfidf_matrix
    except FileNotFoundError:
        raise FileNotFoundError("Make sure to save the data.")

books, tfidf, tfidf_matrix = load_model_data()

# serve covers over https so they load on the https frontend
books['thumbnail'] = books['thumbnail'].str.replace('http://', 'https://', n=1, regex=False)

# app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# models
class BookRequest(BaseModel):
    title: str

class RecommendedBook(BaseModel):
    title: str
    authors: str      
    thumbnail: str    

# api endpoints
@app.get('/')
def root():
    return {"message": "Welcome to the Book Recommendation API!"}

@app.post("/recommend_book", response_model=list[RecommendedBook])
def recommend_book(request: BookRequest):
    title = ' '.join(request.title.split())
    
    book_index = find_book_in_database(title)
    if book_index is not None:
        return get_recommendations_from_database(book_index)
        
    return get_recommendations_from_api(title)

# database recommendations
def find_book_in_database(title):
    book_rows = books[books['title'] == title.lower().strip()]
    if book_rows.empty:
        return None

    # a title can appear several times, so use the copy with the most informative description
    best_index = max(book_rows.index, key=lambda i: tfidf_matrix[i].nnz)
    if tfidf_matrix[best_index].nnz == 0:
        print(f"Book '{title}' found in database but has no description.")
        return None

    print(f"Book '{title}' found in database.")
    return best_index

def get_recommendations_from_database(book_index):
    print("Generating recommendations from database...")
    
    # computing one row on demand instead of keeping the full 7k x 7k similarity matrix in memory
    book_similarities = cosine_similarity(tfidf_matrix[book_index], tfidf_matrix).flatten()
    return top_recommendations(book_similarities, exclude_title=books.iloc[book_index]['title'])


# api based recommendations
def get_recommendations_from_api(title):
    print(f"Book '{title}' not found in database. Fetching from API...")
    
    book_data = fetch_book_data_from_google(title)
    if not book_data:
        raise create_not_found_exception()
    
    if not is_valid_book_match(title, book_data['title']):
        raise create_not_found_exception()
    
    recommended_books = generate_recommendations_for_new_book(book_data)

    if not recommended_books:
        raise create_not_found_exception()
    
    return recommended_books

def fetch_book_data_from_google(title):
    try:
        encoded_title = quote(title)
        url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_title}&maxResults=1"
        if GOOGLE_BOOKS_API_KEY:
            url += f"&key={GOOGLE_BOOKS_API_KEY}"
        
        response = requests.get(url, timeout=5)
        data = response.json()

        if response.status_code != 200:
            print(f"Google Books API error {response.status_code}: {data.get('error', {}).get('message')}")
            return None
        
        if data.get('totalItems', 0) == 0:
            return None
        
        book_info = data['items'][0]['volumeInfo']
        print(book_info)
        return {
            'title': book_info.get('title', ''),  
            'description': book_info.get('description', ''),  
        }
    except Exception as err:
        print(f"Error fetching book data: {err}")
        return None

def generate_recommendations_for_new_book(book_data):
    text_content = book_data['description'].strip()
    
    if not text_content:
        return []
    
    new_book_tfidf = tfidf.transform([text_content])
    similarities_new = cosine_similarity(new_book_tfidf, tfidf_matrix).flatten()
    return top_recommendations(similarities_new, exclude_title=book_data['title'].lower())


# utility functions
def top_recommendations(similarity_scores, exclude_title, count=5):
    # skips the searched book itself and duplicate copies of the same title
    recommended_books = []
    seen_titles = {exclude_title}
    for idx in similarity_scores.argsort()[::-1]:
        if similarity_scores[idx] <= 0:
            break
        book = books.iloc[idx]
        if book['title'] in seen_titles:
            continue
        seen_titles.add(book['title'])
        recommended_books.append(book[['title', 'authors', 'thumbnail']].to_dict())
        if len(recommended_books) == count:
            break
    return recommended_books

def is_valid_book_match(search_title, api_title, threshold= 0.6):
    title_similarity = SequenceMatcher(None, search_title.lower(), api_title.lower()).ratio()
    keyword_match = search_title.lower() in api_title.lower()
    return title_similarity >= threshold or keyword_match

def create_not_found_exception():
    return HTTPException(
        status_code=404,
        detail="Haven't heard of that one! Sorry can't give any recommendations."
    )