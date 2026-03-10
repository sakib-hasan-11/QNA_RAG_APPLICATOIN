# RAG (Retrieval-Augmented Generation) System

A production-ready RAG system for intelligent document querying using OpenAI embeddings, Pinecone vector database, and GPT-4 for answer generation. Fully containerized with automated CI/CD deployment to AWS ECS.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          GitHub Repository                          │
│  Push to main → Automated Testing → Build & Push Docker Images      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       AWS Elastic Container Registry                │
│  • rag-pipeline:latest       (Data processing container)            │
│  • rag-images-container:api-latest       (FastAPI backend)          │
│  • rag-images-container:streamlit-latest (Web UI)                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      AWS Elastic Container Service                  │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐      │
│  │ Pipeline Task   │  │  API Service │  │ Streamlit Service  │      │
│  │ (One-time run)  │  │  (Always on) │  │  (Always on)       │      │
│  │                 │  │  Port 8000   │  │  Port 8501         │      │
│  └─────────────────┘  └──────────────┘  └────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
           │                      │                    │
           ↓                      ↓                    ↓
    ┌───────────┐          ┌───────────┐        ┌──────────┐
    │ S3 Bucket │          │  Pinecone │        │   User   │
    │ (PDFs)    │          │ (Vectors) │        │ Browser  │
    └───────────┘          └───────────┘        └──────────┘
```

## ✨ Features

- **Document Processing**: Load PDFs from S3 or local storage and chunk intelligently
- **Vector Embeddings**: OpenAI text-embedding-3-small for high-quality embeddings
- **Vector Database**: Pinecone for efficient similarity search
- **LLM Integration**: GPT-4 for accurate answer generation
- **RESTful API**: FastAPI backend with health checks and query endpoints
- **Web Interface**: Streamlit UI for easy interaction
- **CI/CD Pipeline**: Automated testing and deployment via GitHub Actions
- **Container Orchestration**: Fully containerized with Docker and AWS ECS
- **Comprehensive Testing**: Unit tests for all core components

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for containerization)
- AWS Account (for ECS deployment)
- OpenAI API Key
- Pinecone Account

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   # Create .env file
   OPENAI_API_KEY=your-openai-key
   PINECONE_API_KEY=your-pinecone-key
   PINECONE_HOST=your-pinecone-host
   ```

4. **Run the indexing pipeline**
   ```bash
   # For local PDFs
   python -c "from src.pipelines.pipeline import RAGPipeline; \
              pipeline = RAGPipeline(); \
              pipeline.indexing_pipeline(data_dir='./data', use_s3=False)"
   
   # For S3 PDFs (requires S3_BUCKET_NAME and S3_FILE_KEY env vars)
   python run_pipeline.py
   ```

5. **Start the API server**
   ```bash
   python api.py
   # API available at http://localhost:8000
   ```

6. **Launch the Streamlit UI**
   ```bash
   streamlit run app.py
   # Web UI available at http://localhost:8501
   ```

## 🐳 Docker Deployment

### Build Images Locally

```bash
# Build API image
docker build -f Dockerfile.api -t rag-api:latest .

# Build Streamlit image
docker build -f Dockerfile.streamlit -t rag-streamlit:latest .

# Build pipeline image
docker build -f Dockerfile.pipeline -t rag-pipeline:latest .
```

### Run Containers

```bash
# Run API
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your-key \
  -e PINECONE_API_KEY=your-key \
  -e PINECONE_HOST=your-host \
  rag-api:latest

# Run Streamlit
docker run -p 8501:8501 \
  -e API_URL=http://localhost:8000 \
  rag-streamlit:latest
```

## ☁️ AWS ECS Deployment

### Setup

1. **Create ECR Repositories**
   ```bash
   aws ecr create-repository --repository-name rag-pipeline --region us-east-1
   aws ecr create-repository --repository-name rag-images-container --region us-east-1
   ```

2. **Configure GitHub Secrets**
   
   Add the following secrets in your GitHub repository settings:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_HOST`

3. **Push to Main Branch**
   ```bash
   git push origin main
   ```
   
   GitHub Actions will automatically:
   - Run all tests
   - Build Docker images
   - Push to ECR:
     - `rag-pipeline:latest` (data processing)
     - `rag-images-container:api-latest` (API service)
     - `rag-images-container:streamlit-latest` (Streamlit UI)

### Deploy on ECS

1. **Create ECS Cluster**
   ```bash
   aws ecs create-cluster --cluster-name rag-cluster
   ```

2. **Run Pipeline Task** (one-time data processing)
   
   Create task definition using `rag-pipeline:latest` with environment variables:
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_HOST`
   - `S3_BUCKET_NAME`
   - `S3_FILE_KEY`
   - `AWS_REGION`

3. **Deploy API Service**
   
   Create task definition using `rag-images-container:api-latest`:
   - Port: 8000
   - Environment: OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_HOST

4. **Deploy Streamlit Service**
   
   Create task definition using `rag-images-container:streamlit-latest`:
   - Port: 8501
   - Environment: `API_URL=http://<api-service-url>:8000`

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Response:
```json
{
  "status": "healthy",
  "timestamp": "2026-03-10T12:00:00Z"
}
```

### Query RAG System
```bash
POST /query
Content-Type: application/json

{
  "query": "What is the main topic of the document?",
  "k": 3
}
```
Response:
```json
{
  "answer": "The document discusses...",
  "sources": ["chunk_1", "chunk_2", "chunk_3"],
  "retrieved_count": 3
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Yes | - |
| `PINECONE_API_KEY` | Pinecone API key | Yes | - |
| `PINECONE_HOST` | Pinecone host URL | Yes | - |
| `S3_BUCKET_NAME` | S3 bucket for PDFs | For S3 mode | - |
| `S3_FILE_KEY` | S3 file path | For S3 mode | - |
| `AWS_REGION` | AWS region | No | `us-east-1` |
| `EMBED_MODEL` | Embedding model | No | `text-embedding-3-small` |
| `LLM_MODEL` | LLM model | No | `gpt-4o-mini` |
| `CHUNK_SIZE` | Document chunk size | No | `800` |
| `CHUNK_OVERLAP` | Chunk overlap | No | `80` |
| `BATCH_SIZE` | Vector upsert batch | No | `100` |
| `API_URL` | API endpoint (Streamlit) | Yes | - |

## 📁 Project Structure

```
rag/
├── .github/workflows/
│   └── cd.yml                 # CI/CD pipeline
├── src/
│   ├── pipelines/
│   │   └── pipeline.py        # Main RAG pipeline
│   ├── scripts/
│   │   ├── chunks.py          # Document chunking
│   │   ├── doc_reader.py      # PDF reader (S3/local)
│   │   ├── load_model.py      # Model loaders
│   │   ├── prompt.py          # Prompt templates
│   │   ├── rag_query.py       # Query execution
│   │   ├── retrieve.py        # Vector retrieval
│   │   └── vector_sender.py   # Pinecone upserter
│   └── utils/
│       ├── log.py             # Logging utilities
│       └── pinecone_DB.py     # Pinecone connection
├── tests/
│   ├── test_*.py              # Unit tests
│   └── conftest.py            # Test configuration
├── api.py                     # FastAPI application
├── app.py                     # Streamlit UI
├── run_pipeline.py            # Pipeline runner for ECS
├── Dockerfile.api             # API container
├── Dockerfile.streamlit       # Streamlit container
├── Dockerfile.pipeline        # Pipeline container
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🧪 Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test files:
```bash
pytest tests/test_chunks.py -v
pytest tests/test_doc_reader.py -v
pytest tests/test_pipeline.py -v
```

With coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cd.yml`) automatically:

1. **Test Stage**
   - Runs on pull requests and pushes
   - Executes unit tests
   - Validates imports and basic functionality

2. **Build & Push Stage** (main branch only)
   - Builds pipeline container → `rag-pipeline:latest`
   - Builds API container → `rag-images-container:api-latest`
   - Builds Streamlit container → `rag-images-container:streamlit-latest`
   - Pushes all images to AWS ECR

3. **Notification Stage**
   - Reports deployment status
   - Provides next steps for ECS service updates

## 📝 Usage Examples

### Python API

```python
from src.pipelines.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Index documents
pipeline.indexing_pipeline(
    data_dir='./data',
    chunk_size=800,
    chunk_overlap=80,
    batch_size=100,
    use_s3=False
)

# Query
result = pipeline.query_pipeline(
    query="What are the key findings?",
    k=3,
    verbose=True
)

print(result['answer'])
print(result['sources'])
```

### REST API

```python
import requests

# Query the system
response = requests.post(
    'http://localhost:8000/query',
    json={
        'query': 'Explain the methodology',
        'k': 5
    }
)

data = response.json()
print(data['answer'])
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: Pipeline fails to load from S3
- **Solution**: Verify `S3_BUCKET_NAME` and `S3_FILE_KEY` are correct and ECS task has S3 read permissions

**Issue**: Pinecone connection fails
- **Solution**: Check `PINECONE_API_KEY` and `PINECONE_HOST` are valid

**Issue**: Docker images not in ECR
- **Solution**: Ensure GitHub Actions completed successfully and check AWS credentials

**Issue**: API returns 500 errors
- **Solution**: Check CloudWatch logs for detailed error messages

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Support

For issues and questions, please open an issue in the GitHub repository.

---

**Built with** ❤️ **using OpenAI, Pinecone, FastAPI, Streamlit, and AWS ECS**
