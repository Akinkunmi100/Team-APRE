# Ultimate AI Review Engine - Service Shutdown Script

function Write-Header {
    param($Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

Write-Header "Stopping Services"

# Stop Python processes
$processes = @(
    "uvicorn",
    "flask",
    "streamlit",
    "celery"
)

foreach ($process in $processes) {
    $running = Get-Process -Name $process -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "Stopping $process..." -ForegroundColor Yellow
        Stop-Process -Name $process -Force
    }
}

# Stop Redis if it was started by us
$redis = Get-Service -Name Redis -ErrorAction SilentlyContinue
if ($redis -and $redis.Status -eq "Running") {
    Write-Host "Stopping Redis service..." -ForegroundColor Yellow
    Stop-Service Redis
}

Write-Header "Cleanup Complete"
Write-Host "All services have been stopped.`n"