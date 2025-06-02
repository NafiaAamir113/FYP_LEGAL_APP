import streamlit as st
import requests
import pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder
import datetime

# Streamlit page setup
st.set_page_config(page_title="LEGAL ASSISTANT", layout="wide")

# Load secrets
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
TOGETHER_AI_API_KEY = st.secrets["TOGETHER_AI_API_KEY"]

# Pinecone setup
INDEX_NAME = "lawdata-index"
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    st.error(f"Index '{INDEX_NAME}' not found.")
    st.stop()

# Initialize Pinecone index
index = pc.Index(INDEX_NAME)

# Load embedding models
embedding_model = SentenceTransformer("BAAI/bge-large-en")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Page Title
st.title("⚖️ LEGAL ASSISTANT")

# Short App Description
st.markdown("This AI-powered legal assistant retrieves relevant legal documents and provides accurate responses to your legal queries.")

# Input field
query = st.text_input("Enter your legal question:")

# Generate Answer Button
if st.button("Generate Answer"):
    if not query:
        st.warning("Please enter a legal question before generating an answer.")
        st.stop()

    if len(query.split()) < 4:
        st.warning("Your query seems incomplete. Please provide more details.")
        st.stop()

    with st.spinner("Searching legal documents..."):
        try:
            query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()
        except Exception as e:
            st.error(f"Embedding generation failed: {e}")
            st.stop()

        try:
            search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        except Exception as e:
            st.error(f"Pinecone query failed: {e}")
            st.stop()

        matches = search_results.get("matches", [])
        if not matches:
            st.warning("No relevant documents found.")
            st.stop()

        # Extract and rerank context
        context_chunks = [match["metadata"]["text"] for match in matches]
        rerank_scores = reranker.predict([(query, chunk) for chunk in context_chunks])
        ranked_results = sorted(zip(context_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
        context_text = "\n\n".join([chunk for chunk, _ in ranked_results[:5]])

        # Construct prompt
        prompt = f"""You are a legal assistant. Given the retrieved legal documents, provide a detailed answer.

Context:
{context_text}

Question: {query}

Answer:"""

        # Query Together AI
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
            st.error(f"AI generation failed: {e}")
            st.stop()

        st.success("AI Response:")
        st.write(answer)

        # Report download
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Legal_Report_{timestamp}.txt"
        report_content = f"LEGAL REPORT\n\nQuestion:\n{query}\n\nAnswer:\n{answer}"

        st.download_button(
            label="📄 Download Legal Report",
            data=report_content,
            file_name=filename,
            mime="text/plain"
        )

# Footer
st.markdown("<p style='text-align: center;'>🚀 Built with Streamlit</p>", unsafe_allow_html=True)
