#!/bin/bash

# AI-Powered Phone Review Engine - Deployment Script
# Supports AWS, Google Cloud, Azure, and Kubernetes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="ai-review-engine"
VERSION=$(git describe --tags --always --dirty)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
ENVIRONMENT=${ENVIRONMENT:-production}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
    fi
    
    # Check cloud CLIs based on deployment target
    case $1 in
        aws)
            if ! command -v aws &> /dev/null; then
                log_error "AWS CLI is not installed"
            fi
            ;;
        gcp)
            if ! command -v gcloud &> /dev/null; then
                log_error "Google Cloud SDK is not installed"
            fi
            ;;
        azure)
            if ! command -v az &> /dev/null; then
                log_error "Azure CLI is not installed"
            fi
            ;;
        k8s)
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is not installed"
            fi
            ;;
    esac
    
    log_success "All prerequisites met"
}

# Build Docker image
build_docker_image() {
    log_info "Building Docker image..."
    
    docker build -t ${PROJECT_NAME}:${VERSION} .
    docker tag ${PROJECT_NAME}:${VERSION} ${PROJECT_NAME}:latest
    
    log_success "Docker image built: ${PROJECT_NAME}:${VERSION}"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    docker run --rm \
        -e DATABASE_URL="postgresql://postgres:testpass@postgres:5432/test_db" \
        -e REDIS_URL="redis://redis:6379" \
        ${PROJECT_NAME}:${VERSION} \
        pytest tests/ -v --cov=./
    
    log_success "All tests passed"
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS..."
    
    # Configure AWS
    AWS_REGION=${AWS_REGION:-us-east-1}
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${ECR_REPOSITORY}
    
    # Create ECR repository if not exists
    aws ecr describe-repositories --repository-names ${PROJECT_NAME} --region ${AWS_REGION} || \
        aws ecr create-repository --repository-name ${PROJECT_NAME} --region ${AWS_REGION}
    
    # Tag and push image
    docker tag ${PROJECT_NAME}:${VERSION} ${ECR_REPOSITORY}:${VERSION}
    docker tag ${PROJECT_NAME}:${VERSION} ${ECR_REPOSITORY}:latest
    docker push ${ECR_REPOSITORY}:${VERSION}
    docker push ${ECR_REPOSITORY}:latest
    
    # Deploy CloudFormation stack
    aws cloudformation deploy \
        --template-file deploy/aws/cloudformation.yaml \
        --stack-name ${PROJECT_NAME}-${ENVIRONMENT} \
        --parameter-overrides \
            EnvironmentName=${ENVIRONMENT} \
            DockerImageURI=${ECR_REPOSITORY}:${VERSION} \
            DBPassword=${DB_PASSWORD} \
            KeyPair=${AWS_KEY_PAIR} \
        --capabilities CAPABILITY_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    # Get outputs
    ALB_URL=$(aws cloudformation describe-stacks \
        --stack-name ${PROJECT_NAME}-${ENVIRONMENT} \
        --query "Stacks[0].Outputs[?OutputKey=='ApplicationURL'].OutputValue" \
        --output text)
    
    log_success "Deployed to AWS"
    log_info "Application URL: ${ALB_URL}"
}

# Deploy to Google Cloud
deploy_gcp() {
    log_info "Deploying to Google Cloud..."
    
    # Configure GCP
    GCP_PROJECT=${GCP_PROJECT:-$(gcloud config get-value project)}
    GCP_REGION=${GCP_REGION:-us-central1}
    GCR_REPOSITORY="gcr.io/${GCP_PROJECT}/${PROJECT_NAME}"
    
    # Configure Docker for GCR
    gcloud auth configure-docker
    
    # Tag and push image
    docker tag ${PROJECT_NAME}:${VERSION} ${GCR_REPOSITORY}:${VERSION}
    docker tag ${PROJECT_NAME}:${VERSION} ${GCR_REPOSITORY}:latest
    docker push ${GCR_REPOSITORY}:${VERSION}
    docker push ${GCR_REPOSITORY}:latest
    
    # Deploy to Cloud Run
    gcloud run deploy ${PROJECT_NAME} \
        --image ${GCR_REPOSITORY}:${VERSION} \
        --platform managed \
        --region ${GCP_REGION} \
        --allow-unauthenticated \
        --set-env-vars "DATABASE_URL=${DATABASE_URL},REDIS_URL=${REDIS_URL}" \
        --memory 2Gi \
        --cpu 2 \
        --min-instances 1 \
        --max-instances 10 \
        --port 8000
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe ${PROJECT_NAME} \
        --platform managed \
        --region ${GCP_REGION} \
        --format 'value(status.url)')
    
    log_success "Deployed to Google Cloud"
    log_info "Service URL: ${SERVICE_URL}"
}

# Deploy to Azure
deploy_azure() {
    log_info "Deploying to Azure..."
    
    # Configure Azure
    AZURE_RESOURCE_GROUP=${AZURE_RESOURCE_GROUP:-${PROJECT_NAME}-rg}
    AZURE_LOCATION=${AZURE_LOCATION:-eastus}
    ACR_NAME=${ACR_NAME:-${PROJECT_NAME}acr}
    ACR_REPOSITORY="${ACR_NAME}.azurecr.io/${PROJECT_NAME}"
    
    # Create resource group
    az group create --name ${AZURE_RESOURCE_GROUP} --location ${AZURE_LOCATION}
    
    # Create ACR
    az acr create --resource-group ${AZURE_RESOURCE_GROUP} \
        --name ${ACR_NAME} \
        --sku Basic
    
    # Login to ACR
    az acr login --name ${ACR_NAME}
    
    # Tag and push image
    docker tag ${PROJECT_NAME}:${VERSION} ${ACR_REPOSITORY}:${VERSION}
    docker tag ${PROJECT_NAME}:${VERSION} ${ACR_REPOSITORY}:latest
    docker push ${ACR_REPOSITORY}:${VERSION}
    docker push ${ACR_REPOSITORY}:latest
    
    # Create App Service Plan
    az appservice plan create \
        --name ${PROJECT_NAME}-plan \
        --resource-group ${AZURE_RESOURCE_GROUP} \
        --is-linux \
        --sku P1V2
    
    # Create Web App
    az webapp create \
        --resource-group ${AZURE_RESOURCE_GROUP} \
        --plan ${PROJECT_NAME}-plan \
        --name ${PROJECT_NAME}-app \
        --deployment-container-image-name ${ACR_REPOSITORY}:${VERSION}
    
    # Configure Web App
    az webapp config appsettings set \
        --resource-group ${AZURE_RESOURCE_GROUP} \
        --name ${PROJECT_NAME}-app \
        --settings \
            DATABASE_URL="${DATABASE_URL}" \
            REDIS_URL="${REDIS_URL}" \
            WEBSITES_PORT=8000
    
    # Get app URL
    APP_URL=$(az webapp show \
        --resource-group ${AZURE_RESOURCE_GROUP} \
        --name ${PROJECT_NAME}-app \
        --query defaultHostName \
        --output tsv)
    
    log_success "Deployed to Azure"
    log_info "Application URL: https://${APP_URL}"
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."
    
    # Check cluster connection
    kubectl cluster-info &> /dev/null || log_error "Cannot connect to Kubernetes cluster"
    
    # Create namespace
    kubectl create namespace ${PROJECT_NAME} --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f deploy/k8s/deployment.yaml
    
    # Wait for rollout
    kubectl rollout status deployment/api -n ${PROJECT_NAME} --timeout=5m
    kubectl rollout status deployment/streamlit -n ${PROJECT_NAME} --timeout=5m
    
    # Get ingress URL
    INGRESS_URL=$(kubectl get ingress review-engine-ingress -n ${PROJECT_NAME} \
        -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -z "$INGRESS_URL" ]; then
        INGRESS_URL=$(kubectl get ingress review-engine-ingress -n ${PROJECT_NAME} \
            -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    fi
    
    log_success "Deployed to Kubernetes"
    log_info "Application URL: http://${INGRESS_URL}"
}

# Local deployment with Docker Compose
deploy_local() {
    log_info "Deploying locally with Docker Compose..."
    
    docker-compose up -d
    
    log_success "Local deployment complete"
    log_info "API: http://localhost:8000"
    log_info "Web App: http://localhost:8501"
    log_info "Flower: http://localhost:5555"
}

# Health check
health_check() {
    URL=$1
    log_info "Performing health check..."
    
    for i in {1..30}; do
        if curl -f ${URL}/api/health &> /dev/null; then
            log_success "Health check passed"
            return 0
        fi
        sleep 10
    done
    
    log_error "Health check failed"
}

# Rollback
rollback() {
    log_warning "Rolling back deployment..."
    
    case $1 in
        aws)
            # Rollback ECS service
            aws ecs update-service \
                --cluster ${PROJECT_NAME}-cluster \
                --service ${PROJECT_NAME}-service \
                --task-definition ${PROJECT_NAME}:$(($VERSION - 1))
            ;;
        k8s)
            # Rollback Kubernetes deployment
            kubectl rollout undo deployment/api -n ${PROJECT_NAME}
            kubectl rollout undo deployment/streamlit -n ${PROJECT_NAME}
            ;;
    esac
    
    log_success "Rollback complete"
}

# Main deployment flow
main() {
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║     AI-Powered Phone Review Engine Deployment       ║"
    echo "║                 Version: ${VERSION}                 ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    DEPLOYMENT_TARGET=${1:-local}
    
    # Check prerequisites
    check_prerequisites ${DEPLOYMENT_TARGET}
    
    # Build Docker image
    build_docker_image
    
    # Run tests
    if [ "${SKIP_TESTS}" != "true" ]; then
        run_tests
    fi
    
    # Deploy based on target
    case ${DEPLOYMENT_TARGET} in
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
        k8s)
            deploy_k8s
            ;;
        local)
            deploy_local
            ;;
        all)
            deploy_aws
            deploy_gcp
            deploy_azure
            deploy_k8s
            ;;
        *)
            log_error "Invalid deployment target: ${DEPLOYMENT_TARGET}"
            echo "Usage: $0 [aws|gcp|azure|k8s|local|all]"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║           Deployment Complete! 🚀                    ║"
    echo "║                                                      ║"
    echo "║   Environment: ${ENVIRONMENT}                       ║"
    echo "║   Version: ${VERSION}                               ║"
    echo "║   Target: ${DEPLOYMENT_TARGET}                      ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Run main function
main $@
