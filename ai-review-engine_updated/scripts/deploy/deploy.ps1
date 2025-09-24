# Deployment Script for AI Review Engine
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment,
    
    [switch]$UseDocker,
    [switch]$SkipTests,
    [switch]$SkipBackup,
    [switch]$Force
)

# Error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Import common functions
. "$PSScriptRoot\..\common\functions.ps1"

# Configuration
$config = @{
    development = @{
        db_host = "localhost"
        redis_host = "localhost"
        api_port = 8000
        web_port = 5000
        dashboard_port = 8501
    }
    staging = @{
        db_host = "staging-db"
        redis_host = "staging-redis"
        api_port = 8000
        web_port = 5000
        dashboard_port = 8501
    }
    production = @{
        db_host = "prod-db"
        redis_host = "prod-redis"
        api_port = 8000
        web_port = 5000
        dashboard_port = 8501
    }
}

function Backup-Database {
    Write-Log "Creating database backup..."
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = "backups/db_backup_$timestamp.dump"
    
    try {
        pg_dump -h $config[$Environment].db_host -U postgres -F c -b -v -f $backupPath ai_review_engine
        Write-Log "Database backup created successfully at $backupPath"
    }
    catch {
        Write-Log "Failed to create database backup: $_" "ERROR"
        exit 1
    }
}

function Stop-Services {
    Write-Log "Stopping services..."
    $services = @("AIReviewAPI", "AIReviewWeb", "AIReviewDashboard", "AIReviewWorker")
    
    foreach ($service in $services) {
        $svc = Get-Service $service -ErrorAction SilentlyContinue
        if ($svc) {
            Stop-Service $service
            Write-Log "Stopped $service"
        }
    }
}

function Start-Services {
    Write-Log "Starting services..."
    
    if ($UseDocker) {
        docker-compose -f docker-compose.yml up -d
    }
    else {
        $services = @("AIReviewAPI", "AIReviewWeb", "AIReviewDashboard", "AIReviewWorker")
        foreach ($service in $services) {
            Start-Service $service
            Write-Log "Started $service"
        }
    }
}

function Update-Code {
    Write-Log "Updating application code..."
    
    # Pull latest changes
    git pull origin main
    
    # Install/update dependencies
    if (-not $UseDocker) {
        pip install -r requirements.txt
        if ($Environment -eq "development") {
            pip install -r requirements-dev.txt
        }
    }
}

function Update-Database {
    Write-Log "Running database migrations..."
    try {
        flask db upgrade
        Write-Log "Database migrations completed successfully"
    }
    catch {
        Write-Log "Database migration failed: $_" "ERROR"
        if (-not $Force) {
            exit 1
        }
    }
}

function Test-Application {
    if (-not $SkipTests) {
        Write-Log "Running tests..."
        pytest tests/
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Tests failed!" "ERROR"
            if (-not $Force) {
                exit 1
            }
        }
    }
}

function Test-Deployment {
    Write-Log "Testing deployment..."
    $endpoints = @(
        "http://localhost:$($config[$Environment].api_port)/health",
        "http://localhost:$($config[$Environment].web_port)/health",
        "http://localhost:$($config[$Environment].dashboard_port)/healthz"
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-Log "Health check passed for $endpoint"
            }
            else {
                Write-Log "Health check failed for $endpoint: $($response.StatusCode)" "WARNING"
            }
        }
        catch {
            Write-Log "Health check failed for $endpoint: $_" "ERROR"
            if (-not $Force) {
                exit 1
            }
        }
    }
}

# Main deployment process
try {
    Write-Log "Starting deployment to $Environment environment"
    
    # Create backup if not skipped
    if (-not $SkipBackup) {
        Backup-Database
    }
    
    # Stop services
    Stop-Services
    
    # Update application code
    Update-Code
    
    # Update database
    Update-Database
    
    # Run tests
    Test-Application
    
    # Start services
    Start-Services
    
    # Test deployment
    Test-Deployment
    
    Write-Log "Deployment completed successfully!"
}
catch {
    Write-Log "Deployment failed: $_" "ERROR"
    if (-not $Force) {
        # Attempt to restore from backup if available
        if (-not $SkipBackup) {
            Write-Log "Attempting to restore from backup..."
            # Add restore logic here
        }
        exit 1
    }
}