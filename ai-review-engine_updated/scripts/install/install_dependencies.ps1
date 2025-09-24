# Installation Script for AI Review Engine
param(
    [string]$Environment = "development",
    [switch]$SkipDocker,
    [switch]$SkipPython,
    [switch]$Force
)

# Error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Logging function
function Write-Log {
    param($Message, $Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Level - $Message"
}

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Log "Please run this script as Administrator" "ERROR"
    exit 1
}

# Install Chocolatey if not already installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Log "Installing Chocolatey..."
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    refreshenv
}

# Install Python if not skipped
if (-not $SkipPython) {
    Write-Log "Installing Python 3.10..."
    choco install python310 -y
    refreshenv
    
    # Create virtual environment
    Write-Log "Creating virtual environment..."
    python -m venv .venv
    .\.venv\Scripts\Activate
    
    # Install Python dependencies
    Write-Log "Installing Python dependencies..."
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if ($Environment -eq "development") {
        pip install -r requirements-dev.txt
    }
}

# Install Docker if not skipped
if (-not $SkipDocker) {
    Write-Log "Installing Docker Desktop..."
    choco install docker-desktop -y
    
    Write-Log "Installing Docker Compose..."
    choco install docker-compose -y
    refreshenv
}

# Install PostgreSQL
Write-Log "Installing PostgreSQL..."
choco install postgresql -y
refreshenv

# Install Redis
Write-Log "Installing Redis..."
choco install redis-64 -y
refreshenv

# Install NGINX
Write-Log "Installing NGINX..."
choco install nginx -y
refreshenv

# Create required directories
$directories = @(
    "logs",
    "data",
    "data/processed",
    "data/raw",
    "config/prod",
    "config/dev"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        Write-Log "Creating directory: $dir"
        New-Item -ItemType Directory -Path $dir -Force
    }
}

# Set up services
Write-Log "Setting up services..."

# PostgreSQL Service
$pgService = Get-Service postgresql* -ErrorAction SilentlyContinue
if ($pgService) {
    Write-Log "Starting PostgreSQL service..."
    Start-Service postgresql*
} else {
    Write-Log "PostgreSQL service not found. Please check installation." "WARNING"
}

# Redis Service
$redisService = Get-Service redis -ErrorAction SilentlyContinue
if ($redisService) {
    Write-Log "Starting Redis service..."
    Start-Service redis
} else {
    Write-Log "Redis service not found. Please check installation." "WARNING"
}

# Create database
Write-Log "Creating database..."
$env:PGPASSWORD = "postgres"
createdb -U postgres ai_review_engine

# Initialize database
Write-Log "Initializing database..."
python scripts/init_db.py

Write-Log "Installation completed successfully!"

# Final checks
Write-Log "Running final checks..."
$services = @{
    "PostgreSQL" = "postgresql*"
    "Redis" = "redis"
    "Docker" = "docker*"
}

foreach ($service in $services.GetEnumerator()) {
    $status = Get-Service $service.Value -ErrorAction SilentlyContinue
    if ($status) {
        Write-Log "$($service.Key) Status: $($status.Status)"
    } else {
        Write-Log "$($service.Key) not found" "WARNING"
    }
}

Write-Log "Installation complete! Please check the logs for any warnings or errors."