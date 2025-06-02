import streamlit as st
import requests
import pinecone
from transformers import AutoTokenizer, AutoModel
import torch
import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit setup
st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    st.error(f"Index '{INDEX_NAME}' not found.")
    st.stop()

index = pc.Index(INDEX_NAME)

# Load embedding model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("hafsanaz0076/bge-finetuned")
model = AutoModel.from_pretrained("hafsanaz0076/bge-finetuned")
model = model.to("cpu")  # Ensure compatibility with Streamlit Cloud

# Embedding function
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return normalized[0].detach().cpu().numpy()  # Safe conversion

# UI
st.title("⚖️ LEGAL ASSISTANT")
st.markdown("AI-powered legal assistant that retrieves relevant documents and answers your legal questions.")

# User input
query = st.text_input("Enter your legal question:")

# On submit
if st.button("Generate Answer"):
    if not query or len(query.split()) < 4:
        st.warning("Please enter a detailed legal question.")
        st.stop()

    with st.spinner("Searching legal database..."):
        try:
            query_embedding = get_embedding(query)
        except Exception as e:
            st.error(f"Failed to create embedding: {e}")
            st.stop()

        try:
            results = index.query(vector=query_embedding.tolist(), top_k=8, include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.stop()

        matches = results.get("matches", [])
        if not matches:
            st.warning("No relevant documents found.")
            st.stop()

        chunks = [m["metadata"]["text"] for m in matches]
        embeddings = [get_embedding(chunk) for chunk in chunks]

        # Re-rank using cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        reranked = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
        context_text = "\n\n".join([c[0] for c in reranked[:5]])

        # Prompt to Together AI
        prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

Context:
{context_text}

Question: {query}

Answer:"""

        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {TOGETHER_AI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "meta-llama/Llama-3-70B-Instruct",
                    "messages": [
                        {"role": "system", "content": "You are an expert in legal matters."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2
                }
            )
            answer = response.json().get("choices", [{}])[0].get("message", {}).get("content", "No valid response from AI.")
        except Exception as e:
            st.error(f"AI request failed: {e}")
            st.stop()

        st.success("AI Response:")
        st.write(answer)

        # Download report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Legal_Report_{timestamp}.txt"
        report = f"LEGAL REPORT\n\nQuestion:\n{query}\n\nAnswer:\n{answer}"
        st.download_button("📄 Download Report", data=report, file_name=filename, mime="text/plain")

st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)
