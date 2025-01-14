import os
import pandas as pd
import numpy as np
import faiss
import re
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import streamlit as st

load_dotenv()

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gen_config = {
    "temperature": 0.5,
    "max_output_tokens": 512
}
gemini_model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    generation_config=gen_config
)

preprocessed_data = pd.read_csv('preprocessed_data.csv')
embeddings = np.load('embeddings_all_minilm.npy')

index_path = 'vector_index.faiss'

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)  

def search_similar_texts(query, index, data, top_k=5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]]
    return results, distances[0]

def generate_response(query, results):
    relevant_texts = " ".join(results["text"].tolist())
    prompt = f"Question : {query}\nVoici des informations pertinentes : {relevant_texts}\nGénère une réponse basée sur ces informations."
    response = gemini_model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")

col1, col2 = st.columns([1, 1])  

with col1:
    st.image("chat.png", width=200)  

with col2:
    st.image("logo.png", width=200)  

st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #555;
        }
        .response {
            background-color: #f1f1f1;
            border-radius: 5px;
            padding: 15px;
            font-size: 16px;
        }
        .input-box {
            margin: 20px 0;
            font-size: 16px;
            padding: 10px;
            width: 100%;
            max-width: 600px;
            border-radius: 5px;
            border: 1px solid #ccc;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">OrientAI</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Permets-moi de t\'aider pour ton orientation.</div>', unsafe_allow_html=True)

query = st.text_input("Pose ta question :", "", key="query", placeholder="Entrez votre question ici...", help="Posez une question sur l'orientation.")

if query:
    with st.spinner('Recherche en cours...'):
        results, distances = search_similar_texts(query, index, preprocessed_data)

        response = generate_response(query, results)

        st.markdown(f'<div class="response"><strong>OrientAI :</strong><br>{response}</div>', unsafe_allow_html=True)
