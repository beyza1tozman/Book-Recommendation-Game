from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
import requests
from scipy.sparse import load_npz
from difflib import SequenceMatcher

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# optional, but without a key requests share Google's anonymous quota, which often runs out
GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY", "").strip() or None
print(f"Google Books API key {'loaded' if GOOGLE_BOOKS_API_KEY else 'NOT set'}.")

class BookLookupUnavailable(Exception):
    """An external book API failed, as opposed to finding no such book."""

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
    
    book_data = fetch_book_data(title)
    if not book_data:
        raise create_not_found_exception()
    
    if not is_valid_book_match(title, book_data['title']):
        raise create_not_found_exception()
    
    recommended_books = generate_recommendations_for_new_book(book_data)

    if not recommended_books:
        raise create_not_found_exception()
    
    return recommended_books

def fetch_book_data(title):
    # Google Books first, Open Library as a keyless fallback when Google is down or out of quota
    google_failed = False
    try:
        book_data = fetch_book_data_from_google(title)
        if book_data and book_data['description'].strip():
            return book_data
    except BookLookupUnavailable as err:
        print(f"Google Books unavailable: {err}")
        google_failed = True

    try:
        return fetch_book_data_from_open_library(title)
    except BookLookupUnavailable as err:
        print(f"Open Library unavailable: {err}")
        if google_failed:
            raise HTTPException(
                status_code=503,
                detail="Book lookup is temporarily unavailable. Please try again in a moment."
            )
        return None

def fetch_book_data_from_google(title):
    params = {'q': title, 'maxResults': 1}
    if GOOGLE_BOOKS_API_KEY:
        params['key'] = GOOGLE_BOOKS_API_KEY

    # Google Books returns sporadic 503s, so retry once before giving up
    for attempt in range(2):
        try:
            response = requests.get("https://www.googleapis.com/books/v1/volumes", params=params, timeout=5)
        except requests.RequestException as err:
            raise BookLookupUnavailable(err)
        if response.status_code == 200:
            break
        if response.status_code < 500 or attempt == 1:
            raise BookLookupUnavailable(f"{response.status_code} {response.text[:300]}")

    data = response.json()
    if data.get('totalItems', 0) == 0:
        return None

    book_info = data['items'][0]['volumeInfo']
    return {
        'title': book_info.get('title', ''),
        'description': book_info.get('description', ''),
    }

def fetch_book_data_from_open_library(title):
    try:
        search = requests.get(
            "https://openlibrary.org/search.json",
            params={'q': title, 'limit': 1, 'fields': 'key,title'},
            timeout=5,
        )
        search.raise_for_status()
        docs = search.json().get('docs', [])
        if not docs:
            return None

        work = requests.get(f"https://openlibrary.org{docs[0]['key']}.json", timeout=5)
        work.raise_for_status()
        description = work.json().get('description', '')
    except (requests.RequestException, ValueError) as err:
        raise BookLookupUnavailable(err)

    # descriptions come either as a plain string or as {"type": ..., "value": ...}
    if isinstance(description, dict):
        description = description.get('value', '')
    return {
        'title': docs[0].get('title', ''),
        'description': description,
    }

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