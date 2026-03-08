"""
Streamlit Frontend for RAG Query System
"""

import os

import requests
import streamlit as st

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Query System", page_icon="🔍", layout="wide")

st.title("🔍 RAG Query System")
st.markdown("Ask questions about your indexed documents")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    k_value = st.slider(
        "Number of documents to retrieve", min_value=1, max_value=10, value=3
    )

    st.divider()
    st.markdown("### API Status")

    # Check API health
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("✓ API Connected")
        else:
            st.error("✗ API Error")
    except:
        st.error("✗ API Offline")

    st.divider()
    st.markdown("### About")
    st.info(
        "This system uses RAG (Retrieval-Augmented Generation) to answer questions based on your documents."
    )

# Main query interface
query = st.text_input(
    "Enter your question:", placeholder="What is the main topic of the documents?"
)

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    submit_button = st.button(
        "🚀 Submit Query", type="primary", use_container_width=True
    )
with col2:
    clear_button = st.button("🗑️ Clear", use_container_width=True)

if clear_button:
    st.rerun()

# Handle query submission
if submit_button and query:
    with st.spinner("🔍 Searching and generating answer..."):
        try:
            # Send query to API
            response = requests.post(
                f"{API_URL}/query", json={"query": query, "k": k_value}, timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Display answer
                st.success("✓ Answer generated successfully!")
                st.markdown("### 📌 Answer")
                st.markdown(f"**{result['answer']}**")

                # Display sources
                st.markdown("### 📚 Sources")
                for i, source in enumerate(result["sources"], 1):
                    with st.expander(
                        f"Source {i}: {source['source']} (Page {source['page']})"
                    ):
                        st.write(f"**Relevance Score:** {source['score']:.4f}")
                        st.write(f"**Source:** {source['source']}")
                        st.write(f"**Page:** {source['page']}")
            else:
                st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")

        except requests.exceptions.Timeout:
            st.error("❌ Request timeout. Please try again.")
        except requests.exceptions.ConnectionError:
            st.error(
                "❌ Cannot connect to API. Please check if the backend is running."
            )
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

elif submit_button and not query:
    st.warning("⚠️ Please enter a question first.")

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - What is the main topic of the documents?
    - Can you summarize the key findings?
    - What are the conclusions mentioned?
    - Explain the methodology used.
    """)
