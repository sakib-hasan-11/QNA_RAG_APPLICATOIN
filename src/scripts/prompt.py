
def create_prompt(query, context_docs):
    """
    Create a prompt with context from retrieved documents.
    
    Args:
        query (str): User's question
        context_docs (list): List of retrieved documents with text and metadata
    
    Returns:
        str: Formatted prompt with context
    """
    context = "\n\n".join([f"Context {i+1}:\n{doc['text']}" for i, doc in enumerate(context_docs)])
    # find the most similar vector with the query . 
    # get the text from the metadata of the most similar vectors.
    # arrange text into a proper readable format.
    
    prompt = f"""You are a helpful AI assistant. Answer the question based on the following context.
    If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {query}

    Answer:"""
        
    return prompt