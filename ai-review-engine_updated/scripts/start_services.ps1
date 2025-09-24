# Ultimate AI Review Engine - Service Startup Script
# This script starts all required services for the AI Review Engine

function Write-Header {
    param($Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Test-Port {
    param($Port)
    $result = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $result.TcpTestSucceeded
}

function Wait-ForPort {
    param($Port, $Service, $Timeout = 30)
    Write-Host "Waiting for $Service to be ready on port $Port..." -NoNewline
    $timer = [Diagnostics.Stopwatch]::StartNew()
    while (-not (Test-Port $Port)) {
        if ($timer.Elapsed.TotalSeconds -gt $Timeout) {
            Write-Host "Timeout!" -ForegroundColor Red
            return $false
        }
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline
    }
    Write-Host "Ready!" -ForegroundColor Green
    return $true
}

# Set working directory to script location
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot\..

# Create logs directory if it doesn't exist
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
}

Write-Header "Checking Dependencies"

# Check if Python is installed
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed or not in PATH!" -ForegroundColor Red
    exit 1
}

# Check if Redis is installed and running
$redis = Get-Service -Name Redis -ErrorAction SilentlyContinue
if (-not $redis) {
    Write-Host "Redis service not found. Installing Redis..." -ForegroundColor Yellow
    choco install redis-64 -y
    Start-Service Redis
} elseif ($redis.Status -ne "Running") {
    Write-Host "Starting Redis service..." -ForegroundColor Yellow
    Start-Service Redis
}

# Check if PostgreSQL is installed and running
$postgres = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue
if (-not $postgres) {
    Write-Host "PostgreSQL service not found. Installing PostgreSQL..." -ForegroundColor Yellow
    choco install postgresql -y
    Start-Service postgresql-x64-15
} elseif ($postgres.Status -ne "Running") {
    Write-Host "Starting PostgreSQL service..." -ForegroundColor Yellow
    Start-Service $postgres.Name
}

Write-Header "Starting Services"

# Function to start a Python service
function Start-PythonService {
    param(
        $ServiceName,
        $Command,
        $WorkingDir,
        $Port
    )
    Write-Host "Starting $ServiceName..." -ForegroundColor Yellow
    $logFile = "logs\$($ServiceName.ToLower()).log"
    $processArgs = "-NoExit -Command `"cd '$WorkingDir'; $Command 2>&1 | Tee-Object -FilePath '$logFile'`""
    Start-Process pwsh -ArgumentList $processArgs
    Wait-ForPort -Port $Port -Service $ServiceName
}

# Start FastAPI backend
Start-PythonService -ServiceName "FastAPI" -Command "uvicorn main:app --host 0.0.0.0 --port 8000" -WorkingDir "api" -Port 8000

# Start Flask web interface
Start-PythonService -ServiceName "Flask" -Command "python app.py" -WorkingDir "web" -Port 5000

# Start Streamlit dashboard
Start-PythonService -ServiceName "Streamlit" -Command "streamlit run dashboard.py" -WorkingDir "streamlit" -Port 8501

# Start Celery worker
Write-Host "Starting Celery worker..." -ForegroundColor Yellow
$workerLog = "logs\worker.log"
$workerArgs = "-NoExit -Command `"cd worker; celery -A tasks worker --loglevel=info -P eventlet 2>&1 | Tee-Object -FilePath '../$workerLog'`""
Start-Process pwsh -ArgumentList $workerArgs

Write-Header "Service Status"

# Display service status
Write-Host "`nService URLs:" -ForegroundColor Green
Write-Host "- FastAPI Backend:       http://localhost:8000"
Write-Host "- API Documentation:     http://localhost:8000/docs"
Write-Host "- Web Interface:         http://localhost:5000"
Write-Host "- Analytics Dashboard:   http://localhost:8501"
Write-Host "- Celery Flower:        http://localhost:5555"

Write-Host "`nLog Files:" -ForegroundColor Green
Write-Host "- API Logs:             logs\fastapi.log"
Write-Host "- Web Logs:             logs\flask.log"
Write-Host "- Dashboard Logs:       logs\streamlit.log"
Write-Host "- Worker Logs:          logs\worker.log"

Write-Header "Startup Complete"
Write-Host "Press Ctrl+C in individual windows to stop services`n"