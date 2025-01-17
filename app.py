import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st
import google.generativeai as genai

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

def search_similar_texts(query, index, data, top_k=20):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]]
    return results, distances[0]

def generate_response(query, results):
    relevant_texts = " ".join(results["text"].tolist())
    prompt = f"""
        Tu es un assistant d'orientation fluide et naturel. La personne pose la question suivante : {query}

        Les informations pertinentes trouvées sont : {relevant_texts}

        Réponds de manière conversationnelle, comme si tu expliquais à un étudiant. Ta réponse doit être concise, engageante et naturelle.
    """
    response = gemini_model.generate_content(prompt)
    return response.text

st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
        .title {
            font-size: 32px;
            font-weight: bold;
            color: #2E86C1; 
            text-align: center;
        }
        .description {
            font-size: 18px;
            text-align: center;
            color: #555; 
        }
        .response {
            background-color: #F4F6F7; 
            border-radius: 5px;
            padding: 15px;
            font-size: 16px;
            color: #2C3E50; 
        }
        .input-box {
            margin: 20px 0;
            font-size: 16px;
            padding: 10px;
            width: 100%;
            max-width: 600px;
            border-radius: 5px;
            border: 1px solid #B2BABB; 
        }
        .centered {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        button {
            background-color: #117A65; /* Vert foncé, bien visible */
            color: white; /* Contraste élevé */
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        button:hover {
            background-color: #148F77; /* Une nuance plus claire au survol */
        }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    subcol1, subcol2, subcol3 = st.columns([1, 2, 1])

    with subcol2:
        st.image("chat.png",  width=200)
           
    st.markdown("""
                <p style="text-align: center; color: #4CAF50; font-size: 18px;">
                Je suis OrientAI ton chatbot d'orientation scolaire. </p>
                """, unsafe_allow_html=True)
        


query = st.text_input(
    "Pose ta question :", 
    "", 
    key="query", 
    placeholder="Entrez votre question ici...", 
    help="Posez une question sur l'orientation."
)

if st.button("Envoyer"):
    if query:
        with st.spinner('Donnes moi une minute...'):
            results, distances = search_similar_texts(query, index, preprocessed_data)
            response = generate_response(query, results)
            st.markdown(f'<div class="response"><strong>OrientAI :</strong><br>{response}</div>', unsafe_allow_html=True)
    else:
        st.warning("Veuillez poser une question avant d'envoyer.")
