@echo off
REM Quick deployment script for RLM Application (Windows)
REM Usage: deploy.bat [render|railway|docker|local]

echo RLM Deployment Script
echo ======================
echo.

if "%~1"=="" (
    echo Usage: deploy.bat [render^|railway^|docker^|local]
    echo.
    echo Options:
    echo   render  - Deploy to Render.com
    echo   railway - Deploy to Railway.app
    echo   docker  - Deploy locally with Docker
    echo   local   - Run locally
    exit /b 1
)

set PLATFORM=%~1

if "%PLATFORM%"=="render" (
    echo Deploying to Render.com...
    
    if not exist "render.yaml" (
        echo Error: render.yaml not found
        exit /b 1
    )
    
    echo 1. Ensure you've pushed to GitHub:
    echo    git push origin main
    echo.
    echo 2. Go to https://dashboard.render.com/blueprints
    echo    and connect your repository
    echo.
    echo 3. Add environment variables in the Render dashboard:
    echo    - OPENAI_API_KEY
    echo.
    echo Done! Render will auto-deploy on git push.
    
) else if "%PLATFORM%"=="railway" (
    echo Deploying to Railway.app...
    
    REM Check if Railway CLI is installed
    where railway >nul 2>nul
    if %errorlevel% neq 0 (
        echo Installing Railway CLI...
        npm install -g @railway/cli
    )
    
    REM Check if logged in
    railway whoami >nul 2>nul
    if %errorlevel% neq 0 (
        echo Please login to Railway:
        railway login
    )
    
    REM Initialize project if needed
    if not exist ".railway\config.json" (
        echo Initializing Railway project...
        railway init
    )
    
    REM Set environment variables
    set /p openai_key="Enter OPENAI_API_KEY (leave empty to skip): "
    if not "%openai_key%"=="" (
        railway variables set OPENAI_API_KEY="%openai_key%"
    )
    
    railway variables set RLM_ROOT_PROVIDER=openai
    railway variables set RLM_SUB_PROVIDER=openai
    
    REM Deploy
    echo.
    echo Deploying...
    railway up
    
    echo.
    echo Deployment complete!
    railway open
    
) else if "%PLATFORM%"=="docker" (
    echo Deploying locally with Docker...
    
    REM Check if Docker is installed
    where docker >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: Docker not found. Please install Docker first.
        exit /b 1
    )
    
    REM Check if docker-compose is installed
    where docker-compose >nul 2>nul
    if %errorlevel% neq 0 (
        echo Error: docker-compose not found. Please install docker-compose first.
        exit /b 1
    )
    
    REM Build and run
    echo Building and starting containers...
    docker-compose up --build -d
    
    echo.
    echo Docker deployment complete!
    echo App is running at: http://localhost:8000
    echo.
    echo View logs: docker-compose logs -f
    echo Stop: docker-compose down
    
) else if "%PLATFORM%"=="local" (
    echo Running locally...
    
    REM Check if virtual environment exists
    if not exist "venv" (
        echo Creating virtual environment...
        python -m venv venv
    )
    
    REM Activate virtual environment
    call venv\Scripts\activate
    
    REM Install dependencies
    echo Installing dependencies...
    pip install -r requirements.txt
    
    REM Check for .env file
    if not exist ".env" (
        echo Warning: .env file not found. Using mock mode.
        set RLM_ROOT_PROVIDER=mock
        set RLM_SUB_PROVIDER=mock
    )
    
    REM Create uploads directory
    if not exist "uploads" mkdir uploads
    
    REM Run the app
    echo.
    echo Starting RLM Application...
    echo ================================
    python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
    
) else (
    echo Unknown platform: %PLATFORM%
    echo Usage: deploy.bat [render^|railway^|docker^|local]
    exit /b 1
)
