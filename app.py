import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=API_KEY) 

# --- 1. PAGE SETUP (Minimalist High-Tech) ---
st.set_page_config(
    page_title="Orthodox Apologist Pro", page_icon="☦️", layout="centered"
)

st.title("☦️ Orthodox Theological Apologist")
st.caption("Enterprise-grade RAG engine for precise theological analysis.")

# --- 2. CACHED RESOURCE LOADING (RAM Saver) ---
@st.cache_resource
def init_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists("data/knowledge.txt"):
        with open("data/knowledge.txt", "r", encoding="utf-8") as f:
            # Load text and remove empty lines
            knowledge = [line.strip() for line in f.readlines() if line.strip()]
    else:
        knowledge = [
            "The Nicene Creed states that the Holy Spirit proceeds from the Father."
        ]

    embeddings = embedder.encode(knowledge)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index, knowledge


embedder, index, knowledge = init_resources()

# --- 3. SESSION STATE MEMORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if "translation" in msg:
            st.markdown("### 🇪🇹 Amharic Translation")
            st.write(msg["translation"])

        if "source" in msg:
            st.caption(f"**Verified Source:** {msg['source']}")

# --- 4. CHAT INTERACTION LOGIC ---
if user_input := st.chat_input("Ask a theological or historical question..."):

    # Display user message instantly
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Local Search Step
    query_vec = embedder.encode([user_input])
    _, indices = index.search(np.array(query_vec), k=1)
    matched_context = knowledge[indices[0][0]]

    # System prompt ensuring bank-grade strictness
    system_instruction = f"""
    You are an expert Orthodox Christian apologist. 
    Your primary source of truth is this specific text: "{matched_context}"
    
    CRITICAL RULES:
    1. Answer the question using the provided text.
    2. If the user's question cannot be answered by or is completely unrelated to the provided text, politely respond with: "I am programmed to only discuss verified Orthodox theological documents. I cannot find relevant data for this request."
    3. Do not make up facts. Stay entirely faithful to the source text.
    """

    # Call Cloud API
    with st.chat_message("assistant"):
        with st.spinner("Analyzing sacred texts..."):
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_input,
                config={"system_instruction": system_instruction},
            )

            output_text = response.text

            # --- Translation Step ---
            translation_response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=f"""
    Translate the following text into Amharic.

    IMPORTANT:
    - Return ONLY the translated text.
    - Do NOT explain anything.
    - Do NOT break down phrases.
    - Do NOT add notes.
    - Do NOT add introductions.

    Text:
    {output_text}
    """,
            )
            translated_text = translation_response.text
            st.write(output_text)
            st.markdown("### 🇪🇹 Amharic Translation")
            st.write(translated_text)

            # Save assistant response to memory
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": output_text,
                    "translation": translated_text,
                    "source": matched_context,
                }
            )
