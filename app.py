import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

book_pivot = pd.read_csv('book_pivot.csv').set_index('title')
book_list = book_pivot.index.tolist()
model = pickle.load(open('model.pkl','rb'))

def recommend(book_name):
    book_index = np.where([book_pivot.index==book_name])[1][0]
    distances, suggestions = model.kneighbors(book_pivot.iloc[book_index,:].values.reshape(1,-1),n_neighbors=6)
    
    books=[]
    for i in range(len(suggestions[0])):
        index=suggestions[0][i]
        book = book_pivot.index[index]
        books.append(book)

    return books
 

st.title("Book Recommendation App")

selected_book = st.selectbox('Select any Book',book_list)
btn = st.button('Find the Recommended Books')
if btn:
    recommended_books = recommend(selected_book)
    st.write(recommended_books[1:])
