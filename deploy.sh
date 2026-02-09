#!/bin/bash
# Quick deployment script for RLM Application
# Usage: ./deploy.sh [render|railway|docker|local]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RLM Deployment Script${NC}"
echo "======================"
echo ""

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./deploy.sh [render|railway|docker|local]"
    echo ""
    echo "Options:"
    echo "  render  - Deploy to Render.com"
    echo "  railway - Deploy to Railway.app"
    echo "  docker  - Deploy locally with Docker"
    echo "  local   - Run locally"
    exit 1
fi

PLATFORM=$1

case $PLATFORM in
    render)
        echo -e "${YELLOW}Deploying to Render.com...${NC}"
        
        # Check if render.yaml exists
        if [ ! -f "render.yaml" ]; then
            echo -e "${RED}Error: render.yaml not found${NC}"
            exit 1
        fi
        
        echo "1. Ensure you've pushed to GitHub:"
        echo "   git push origin main"
        echo ""
        echo "2. Go to https://dashboard.render.com/blueprints"
        echo "   and connect your repository"
        echo ""
        echo "3. Add environment variables in the Render dashboard:"
        echo "   - OPENAI_API_KEY"
        echo ""
        echo -e "${GREEN}Done! Render will auto-deploy on git push.${NC}"
        ;;
        
    railway)
        echo -e "${YELLOW}Deploying to Railway.app...${NC}"
        
        # Check if Railway CLI is installed
        if ! command -v railway &> /dev/null; then
            echo "Installing Railway CLI..."
            npm install -g @railway/cli
        fi
        
        # Check if logged in
        if ! railway whoami &> /dev/null; then
            echo "Please login to Railway:"
            railway login
        fi
        
        # Initialize project if needed
        if [ ! -f ".railway/config.json" ]; then
            echo "Initializing Railway project..."
            railway init
        fi
        
        # Set environment variables
        echo ""
        read -p "Enter OPENAI_API_KEY (leave empty to skip): " openai_key
        if [ ! -z "$openai_key" ]; then
            railway variables set OPENAI_API_KEY="$openai_key"
        fi
        
        railway variables set RLM_ROOT_PROVIDER=openai
        railway variables set RLM_SUB_PROVIDER=openai
        
        # Deploy
        echo ""
        echo "Deploying..."
        railway up
        
        echo ""
        echo -e "${GREEN}Deployment complete!${NC}"
        railway open
        ;;
        
    docker)
        echo -e "${YELLOW}Deploying locally with Docker...${NC}"
        
        # Check if Docker is installed
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker not found. Please install Docker first.${NC}"
            exit 1
        fi
        
        # Check if docker-compose is installed
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}Error: docker-compose not found. Please install docker-compose first.${NC}"
            exit 1
        fi
        
        # Check for .env file
        if [ ! -f ".env" ]; then
            echo -e "${YELLOW}Warning: .env file not found. Using .env.example...${NC}"
            if [ -f ".env.example" ]; then
                cp .env.example .env
                echo "Please edit .env and add your API keys"
                exit 1
            fi
        fi
        
        # Build and run
        echo "Building and starting containers..."
        docker-compose up --build -d
        
        echo ""
        echo -e "${GREEN}Docker deployment complete!${NC}"
        echo "App is running at: http://localhost:8000"
        echo ""
        echo "View logs: docker-compose logs -f"
        echo "Stop: docker-compose down"
        ;;
        
    local)
        echo -e "${YELLOW}Running locally...${NC}"
        
        # Check if virtual environment exists
        if [ ! -d "venv" ]; then
            echo "Creating virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Install dependencies
        echo "Installing dependencies..."
        pip install -r requirements.txt
        
        # Check for .env file
        if [ ! -f ".env" ]; then
            echo -e "${YELLOW}Warning: .env file not found. Using mock mode.${NC}"
            export RLM_ROOT_PROVIDER=mock
            export RLM_SUB_PROVIDER=mock
        else
            export $(cat .env | grep -v '^#' | xargs)
        fi
        
        # Create uploads directory
        mkdir -p uploads
        
        # Run the app
        echo ""
        echo -e "${GREEN}Starting RLM Application...${NC}"
        echo "================================"
        python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
        ;;
        
    *)
        echo -e "${RED}Unknown platform: $PLATFORM${NC}"
        echo "Usage: ./deploy.sh [render|railway|docker|local]"
        exit 1
        ;;
esac
