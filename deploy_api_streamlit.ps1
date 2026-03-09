# =============================================================================
# Deploy API and Streamlit Containers to ECR
# =============================================================================
# This script builds FastAPI and Streamlit Docker images and pushes them to ECR
# Run this after the pipeline has successfully completed in ECS
# =============================================================================

# Configuration
$AWS_REGION = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$ECR_REPO_API = if ($env:ECR_REPO_API) { $env:ECR_REPO_API } else { "rag-api" }
$ECR_REPO_STREAMLIT = if ($env:ECR_REPO_STREAMLIT) { $env:ECR_REPO_STREAMLIT } else { "rag-streamlit" }

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Deploying API and Streamlit to ECR" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Check if AWS CLI is installed
if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    Write-Host "Error: AWS CLI is not installed" -ForegroundColor Red
    exit 1
}

# Check if Docker is installed
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Docker is not installed" -ForegroundColor Red
    exit 1
}

# Get AWS Account ID
Write-Host "Getting AWS Account ID..." -ForegroundColor Yellow
try {
    $AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
    Write-Host "✓ AWS Account ID: $AWS_ACCOUNT_ID" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "Error: Failed to get AWS Account ID. Make sure AWS credentials are configured." -ForegroundColor Red
    exit 1
}

# Get ECR Registry
$ECR_REGISTRY = "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
Write-Host "ECR Registry: $ECR_REGISTRY" -ForegroundColor Green
Write-Host ""

# Login to ECR
Write-Host "Logging in to Amazon ECR..." -ForegroundColor Yellow
try {
    $loginCommand = aws ecr get-login-password --region $AWS_REGION
    $loginCommand | docker login --username AWS --password-stdin $ECR_REGISTRY
    Write-Host "✓ Successfully logged in to ECR" -ForegroundColor Green
    Write-Host ""
} catch {
    Write-Host "Error: Failed to login to ECR" -ForegroundColor Red
    exit 1
}

# Generate image tag (using timestamp)
$IMAGE_TAG = "v$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "Image Tag: $IMAGE_TAG" -ForegroundColor Green
Write-Host ""

# =============================================================================
# Build and Push API Container
# =============================================================================
Write-Host "========================================" -ForegroundColor Green
Write-Host "Building API Container" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if (-not (Test-Path "Dockerfile.api")) {
    Write-Host "Error: Dockerfile.api not found" -ForegroundColor Red
    exit 1
}

Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.api -t "$ECR_REGISTRY/${ECR_REPO_API}:$IMAGE_TAG" .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build API image" -ForegroundColor Red
    exit 1
}

docker tag "$ECR_REGISTRY/${ECR_REPO_API}:$IMAGE_TAG" "$ECR_REGISTRY/${ECR_REPO_API}:latest"
Write-Host "✓ API image built successfully" -ForegroundColor Green
Write-Host ""

Write-Host "Pushing API image to ECR..." -ForegroundColor Yellow
docker push "$ECR_REGISTRY/${ECR_REPO_API}:$IMAGE_TAG"
docker push "$ECR_REGISTRY/${ECR_REPO_API}:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to push API image" -ForegroundColor Red
    exit 1
}
Write-Host "✓ API image pushed successfully" -ForegroundColor Green
Write-Host ""

# =============================================================================
# Build and Push Streamlit Container
# =============================================================================
Write-Host "========================================" -ForegroundColor Green
Write-Host "Building Streamlit Container" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

if (-not (Test-Path "Dockerfile.streamlit")) {
    Write-Host "Error: Dockerfile.streamlit not found" -ForegroundColor Red
    exit 1
}

Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -f Dockerfile.streamlit -t "$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:$IMAGE_TAG" .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to build Streamlit image" -ForegroundColor Red
    exit 1
}

docker tag "$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:$IMAGE_TAG" "$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:latest"
Write-Host "✓ Streamlit image built successfully" -ForegroundColor Green
Write-Host ""

Write-Host "Pushing Streamlit image to ECR..." -ForegroundColor Yellow
docker push "$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:$IMAGE_TAG"
docker push "$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:latest"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Failed to push Streamlit image" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Streamlit image pushed successfully" -ForegroundColor Green
Write-Host ""

# =============================================================================
# Summary
# =============================================================================
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ Deployment Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Image URIs:"
Write-Host "  API (tagged):      $ECR_REGISTRY/${ECR_REPO_API}:$IMAGE_TAG"
Write-Host "  API (latest):      $ECR_REGISTRY/${ECR_REPO_API}:latest"
Write-Host "  Streamlit (tagged):$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:$IMAGE_TAG"
Write-Host "  Streamlit (latest):$ECR_REGISTRY/${ECR_REPO_STREAMLIT}:latest"
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host "  1. Update your ECS task definitions with these image URIs"
Write-Host "  2. Deploy the services to ECS"
Write-Host ""
