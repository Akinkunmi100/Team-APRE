# WhatsApp Integration Service Installation Script
param(
    [switch]$Production
)

# Error handling
$ErrorActionPreference = "Stop"

function Write-Log {
    param($Message)
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $Message"
}

# Create virtual environment
Write-Log "Creating virtual environment..."
python -m venv .venv
. .\.venv\Scripts\Activate

# Install dependencies
Write-Log "Installing dependencies..."
pip install -r requirements.txt
if (-not $Production) {
    pip install -r requirements-dev.txt
}

# Create environment file
if (-not (Test-Path .env)) {
    Write-Log "Creating environment file..."
    Copy-Item .env.example .env
    Write-Log "Please update the .env file with your WhatsApp API credentials"
}

# Create logs directory
if (-not (Test-Path logs)) {
    Write-Log "Creating logs directory..."
    New-Item -ItemType Directory -Path logs
}

# Install as Windows service
if ($Production) {
    Write-Log "Installing as Windows service..."
    # Install NSSM if not already installed
    if (-not (Get-Command nssm -ErrorAction SilentlyContinue)) {
        choco install nssm -y
    }

    # Create service
    $serviceName = "WhatsAppIntegration"
    $pythonPath = "$PSScriptRoot\.venv\Scripts\python.exe"
    $scriptPath = "$PSScriptRoot\main.py"
    
    nssm install $serviceName $pythonPath $scriptPath
    nssm set $serviceName AppDirectory $PSScriptRoot
    nssm set $serviceName DisplayName "WhatsApp Integration Service"
    nssm set $serviceName Description "WhatsApp integration service for AI Review Engine"
    nssm set $serviceName AppEnvironment "PATH=$PSScriptRoot\.venv\Scripts;$env:PATH"
    
    # Start service
    Start-Service $serviceName
    Write-Log "Service installed and started successfully"
}

Write-Log "Installation completed successfully!"
Write-Log "Next steps:"
Write-Log "1. Update the .env file with your WhatsApp API credentials"
Write-Log "2. Run 'python main.py' to start the service in development mode"
Write-Log "3. Configure your WhatsApp Business API webhook to point to your service"