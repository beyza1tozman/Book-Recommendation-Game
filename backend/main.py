from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from urllib.parse import quote
from scipy.sparse import load_npz
from difflib import SequenceMatcher

# load data
def load_model_data():
    try:
        books = pd.read_pickle("books.pkl")
        similarities = np.load("similarities.npy")
        with open("tfidf_vectorizer.pkl", "rb") as f:
            tfidf = pickle.load(f)
        tfidf_matrix = load_npz("tfidf_matrix.npz")
        return books, similarities, tfidf, tfidf_matrix 
    except FileNotFoundError:
        raise FileNotFoundError("Make sure to save the data.")

books, similarities, tfidf, tfidf_matrix = load_model_data()

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
    book_row = books[books['title'] == title.lower().strip()]
    if not book_row.empty:
        print(f"Book '{title}' found in database.")
        return book_row.index[0]
    
    return None

def get_recommendations_from_database(book_index):
    print("Generating recommendations from database...")
    
    distances = sorted(
        enumerate(similarities[book_index]), 
        reverse=True, 
        key=lambda x: x[1]
    )
    
    return [books.iloc[i][['title', 'authors', 'thumbnail']].to_dict() for i, _ in distances[1:6]]


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
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
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
    top_indices = similarities_new.argsort()[-5:][::-1]
    
    return [books.iloc[idx][['title', 'authors', 'thumbnail']].to_dict() 
            for idx in top_indices]


# utility functions
def is_valid_book_match(search_title, api_title, threshold= 0.6):
    title_similarity = SequenceMatcher(None, search_title.lower(), api_title.lower()).ratio()
    keyword_match = search_title.lower() in api_title.lower()
    return title_similarity >= threshold or keyword_match

def create_not_found_exception():
    return HTTPException(
        status_code=404,
        detail="Haven't heard of that one! Sorry can't give any recommendations."
    )