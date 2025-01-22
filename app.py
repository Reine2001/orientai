import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import streamlit as st
from streamlit_chat import message
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

preprocessed_data = pd.read_csv('original_texts_combines.csv')
embeddings = np.load('embeddings_combines.npy')

index_path = 'index_faiss_combines.faiss'
if os.path.exists(index_path):
    index = faiss.read_index(index_path)
else:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)

def search_similar_texts(query, index, data, top_k=5):
   
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    results = data.iloc[indices[0]].reset_index(drop=True)
    return results, distances[0]

def generate_response(query, results):

    relevant_texts = " ".join(results["text"].tolist())
    prompt = f"""
        Tu es un assistant d'orientation fluide et naturel. La personne pose la question suivante : {query}

        Les informations pertinentes trouvées sont : {relevant_texts}

        Réponds de manière conversationnelle, comme si tu expliquais à un étudiant. Ta réponse doit être concise, engageante et naturelle.
    """
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

st.set_page_config(page_title="OrientAI", page_icon="🎓", layout="centered")

st.markdown(
    """
    <style>
    .response {
        background-color: #F4F6F7;
        border-radius: 5px;
        padding: 15px;
        font-size: 16px;
        color: #2C3E50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎓 Chatbot d'orientation scolaire")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    else:
        message(msg["content"], is_user=False)

st.write("---")  

query = st.text_input(
    "Permets moi de t'aider à t'orienter :",
    "",
    placeholder="Entres ta question ici..."
)
send_button = st.button("Envoyer")

if send_button and query.strip():
    st.session_state["messages"].append({"role": "user", "content": query})

    with st.spinner("Donnes moi une minute..."):
        try:
            results, distances = search_similar_texts(query, index, preprocessed_data)
            
            response = generate_response(query, results)
            
            st.session_state["messages"].append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Erreur lors de la génération de la réponse : {e}")

    st.rerun()
