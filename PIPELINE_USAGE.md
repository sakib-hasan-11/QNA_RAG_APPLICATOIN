# RAG Pipeline - Usage Guide

## Overview

This RAG (Retrieval-Augmented Generation) pipeline provides a complete solution for:
1. **Indexing documents**: Load PDFs, chunk them, embed, and store in Pinecone
2. **Querying**: Retrieve relevant documents and generate answers using LLM

## Project Structure

```
src/
├── pipelines/
│   └── pipeline.py          # Main pipeline orchestration
├── scripts/
│   ├── doc_reader.py        # PDF document loader
│   ├── chunks.py            # Document chunking
│   ├── load_model.py        # Model loading (embeddings & LLM)
│   ├── vector_sender.py     # Vector embedding and storage
│   ├── retrieve.py          # Document retrieval from Pinecone
│   ├── prompt.py            # Prompt creation
│   └── rag_query.py         # RAG query execution
└── utils/
    ├── log.py               # Logging utility
    └── pinecone_DB.py       # Pinecone database connection
```

## Prerequisites

### Environment Variables

Set the following environment variables:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:PINECONE_API_KEY = "your-pinecone-api-key"
$env:PINECONE_HOST = "your-pinecone-host-url"
```

### Required Packages

Install required packages (should be in requirements.txt):

```bash
pip install langchain langchain-openai langchain-community pinecone-client pypdf tqdm
```

## Usage

### Method 1: Using the Pipeline Class

```python
from src.pipelines.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Index documents
pipeline.indexing_pipeline(
    data_dir="data/",           # Directory containing PDFs
    chunk_size=800,             # Size of each chunk
    chunk_overlap=80,           # Overlap between chunks
    batch_size=100              # Batch size for upserting to Pinecone
)

# Query the indexed documents
result = pipeline.query_pipeline(
    query="What is the main topic of the documents?",
    k=3,                        # Number of relevant docs to retrieve
    verbose=True                # Print intermediate steps
)

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Method 2: Using Convenience Functions

```python
from src.pipelines.pipeline import run_indexing, run_query

# Index documents
run_indexing(data_dir="data/", chunk_size=800, chunk_overlap=80, batch_size=100)

# Query
result = run_query(query="What is the main topic?", k=3, verbose=True)
```

### Method 3: Command Line Interface

#### Indexing Mode

```bash
python src/pipelines/pipeline.py index --data_dir "data/" --chunk_size 800 --chunk_overlap 80 --batch_size 100
```

#### Query Mode

```bash
python src/pipelines/pipeline.py query --query "What is the main topic?" --k 3 --verbose
```

## Pipeline Workflow

### Indexing Pipeline

1. **Load Models**: Initialize embedding model and LLM
2. **Connect to Database**: Establish Pinecone connection
3. **Load Documents**: Read PDFs from specified directory
4. **Chunk Documents**: Split documents into smaller chunks
5. **Embed & Store**: Generate embeddings and upload to Pinecone

### Query Pipeline

1. **Load Models**: Initialize embedding model and LLM (if not already loaded)
2. **Connect to Database**: Establish Pinecone connection (if not already connected)
3. **Retrieve Documents**: Find most similar documents using cosine similarity
4. **Create Prompt**: Format query with retrieved context
5. **Generate Answer**: Use LLM to generate response

## Configuration Options

### Chunking Parameters

- `chunk_size` (default: 800): Maximum size of each text chunk
- `chunk_overlap` (default: 80): Number of characters to overlap between chunks
  - Larger overlap helps maintain context across chunks
  - Recommended: 10-15% of chunk_size

### Model Parameters

- `embed_model_name` (default: 'text-embedding-3-small'): OpenAI embedding model
- `llm_model_name` (default: 'gpt-4o-mini'): OpenAI language model

Available embedding models:
- `text-embedding-3-small` (faster, lower cost)
- `text-embedding-3-large` (higher quality)

Available LLM models:
- `gpt-4o-mini` (fast, cost-effective)
- `gpt-4o` (higher quality)
- `gpt-4-turbo` (balanced)

### Retrieval Parameters

- `k` (default: 3): Number of most similar documents to retrieve
  - Lower values: More focused, faster
  - Higher values: More comprehensive context

### Batch Parameters

- `batch_size` (default: 100): Number of vectors to batch before upserting to Pinecone
  - Larger batches: Faster indexing but more memory usage
  - Recommended: 50-200

## Example Workflows

### Complete Example: Index and Query

```python
from src.pipelines.pipeline import RAGPipeline
import os

# Set environment variables
os.environ['OPENAI_API_KEY'] = 'your-key-here'
os.environ['PINECONE_API_KEY'] = 'your-key-here'
os.environ['PINECONE_HOST'] = 'your-host-here'

# Initialize pipeline
pipeline = RAGPipeline()

# Step 1: Index your documents
print("Starting indexing...")
pipeline.indexing_pipeline(
    data_dir="data/",
    chunk_size=800,
    chunk_overlap=80,
    batch_size=100
)
print("Indexing complete!")

# Step 2: Query the indexed documents
queries = [
    "What is the main topic?",
    "What are the key findings?",
    "Can you summarize the conclusions?"
]

for query in queries:
    result = pipeline.query_pipeline(query=query, k=5, verbose=True)
    print(f"\nQ: {query}")
    print(f"A: {result['answer']}")
    print(f"Sources: {[s['source'] for s in result['sources']]}")
    print("-" * 80)
```

### Reusing Initialized Pipeline

```python
# If you've already indexed documents, just initialize and query
pipeline = RAGPipeline()

# Query immediately (models and DB will auto-initialize)
result = pipeline.query_pipeline(
    query="Your question here",
    k=3,
    verbose=True
)
```

## Logging

All operations are logged with detailed information:
- Step-by-step progress
- Success/failure notifications
- Error messages with stack traces

Log format:
```
YYYY-MM-DD HH:MM:SS | LEVEL | module_name | message
```

## Error Handling

All pipeline functions include comprehensive error handling:
- Missing environment variables
- File not found errors
- API connection errors
- Model loading errors

Errors are logged and then re-raised for proper handling at the application level.

## Tips for Best Results

1. **Document Quality**
   - Ensure PDFs are text-based (not scanned images)
   - Remove unnecessary pages or headers/footers if possible

2. **Chunking Strategy**
   - For technical documents: Smaller chunks (500-800 chars)
   - For narrative content: Larger chunks (1000-1500 chars)
   - Always use some overlap (10-15% of chunk size)

3. **Retrieval**
   - Start with k=3-5 for most queries
   - Increase k for complex questions requiring more context
   - Monitor retrieved document quality

4. **Query Formulation**
   - Be specific and clear in your questions
   - Include relevant keywords from your documents
   - Ask one question at a time for best results

## Troubleshooting

### Common Issues

1. **"Missing environment variable" error**
   - Solution: Set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_HOST

2. **"PDF file not found" error**
   - Solution: Check that data_dir path is correct and contains PDF files

3. **"Error connecting to Pinecone" error**
   - Solution: Verify PINECONE_HOST and PINECONE_API_KEY are correct
   - Check that your Pinecone index exists

4. **Import errors**
   - Solution: Ensure all required packages are installed
   - Run from the project root directory

## Performance Considerations

- **Indexing Speed**: Depends on document size and batch_size
  - Typical: ~100-200 chunks per minute
  
- **Query Speed**: Usually 2-5 seconds per query
  - Embedding: ~0.5s
  - Retrieval: ~0.5s
  - LLM generation: 1-4s

## Next Steps

1. Index your documents using the indexing pipeline
2. Test queries to ensure good retrieval quality
3. Adjust chunk_size and k parameters as needed
4. Implement in your application using the provided APIs
