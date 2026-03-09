#!/bin/bash

# =============================================================================
# Deploy API and Streamlit Containers to ECR
# =============================================================================
# This script builds FastAPI and Streamlit Docker images and pushes them to ECR
# Run this after the pipeline has successfully completed in ECS
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPO_API="${ECR_REPO_API:-rag-api}"
ECR_REPO_STREAMLIT="${ECR_REPO_STREAMLIT:-rag-streamlit}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploying API and Streamlit to ECR${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Get AWS Account ID
echo -e "${YELLOW}Getting AWS Account ID...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo -e "${GREEN}✓ AWS Account ID: $AWS_ACCOUNT_ID${NC}"
echo ""

# Get ECR Registry
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
echo -e "${GREEN}ECR Registry: $ECR_REGISTRY${NC}"
echo ""

# Login to ECR
echo -e "${YELLOW}Logging in to Amazon ECR...${NC}"
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
echo -e "${GREEN}✓ Successfully logged in to ECR${NC}"
echo ""

# Generate image tag (using timestamp)
IMAGE_TAG="v$(date +%Y%m%d-%H%M%S)"
echo -e "${GREEN}Image Tag: $IMAGE_TAG${NC}"
echo ""

# =============================================================================
# Build and Push API Container
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building API Container${NC}"
echo -e "${GREEN}========================================${NC}"

if [ ! -f "Dockerfile.api" ]; then
    echo -e "${RED}Error: Dockerfile.api not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f Dockerfile.api -t $ECR_REGISTRY/$ECR_REPO_API:$IMAGE_TAG .
docker tag $ECR_REGISTRY/$ECR_REPO_API:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPO_API:latest
echo -e "${GREEN}✓ API image built successfully${NC}"
echo ""

echo -e "${YELLOW}Pushing API image to ECR...${NC}"
docker push $ECR_REGISTRY/$ECR_REPO_API:$IMAGE_TAG
docker push $ECR_REGISTRY/$ECR_REPO_API:latest
echo -e "${GREEN}✓ API image pushed successfully${NC}"
echo ""

# =============================================================================
# Build and Push Streamlit Container
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Streamlit Container${NC}"
echo -e "${GREEN}========================================${NC}"

if [ ! -f "Dockerfile.streamlit" ]; then
    echo -e "${RED}Error: Dockerfile.streamlit not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Building Docker image...${NC}"
docker build -f Dockerfile.streamlit -t $ECR_REGISTRY/$ECR_REPO_STREAMLIT:$IMAGE_TAG .
docker tag $ECR_REGISTRY/$ECR_REPO_STREAMLIT:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPO_STREAMLIT:latest
echo -e "${GREEN}✓ Streamlit image built successfully${NC}"
echo ""

echo -e "${YELLOW}Pushing Streamlit image to ECR...${NC}"
docker push $ECR_REGISTRY/$ECR_REPO_STREAMLIT:$IMAGE_TAG
docker push $ECR_REGISTRY/$ECR_REPO_STREAMLIT:latest
echo -e "${GREEN}✓ Streamlit image pushed successfully${NC}"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Image URIs:"
echo "  API (tagged):      $ECR_REGISTRY/$ECR_REPO_API:$IMAGE_TAG"
echo "  API (latest):      $ECR_REGISTRY/$ECR_REPO_API:latest"
echo "  Streamlit (tagged):$ECR_REGISTRY/$ECR_REPO_STREAMLIT:$IMAGE_TAG"
echo "  Streamlit (latest):$ECR_REGISTRY/$ECR_REPO_STREAMLIT:latest"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Update your ECS task definitions with these image URIs"
echo "  2. Deploy the services to ECS"
echo ""
