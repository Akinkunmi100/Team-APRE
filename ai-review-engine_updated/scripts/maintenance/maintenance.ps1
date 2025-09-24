# Maintenance Script for AI Review Engine
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("backup", "cleanup", "healthcheck", "logs", "optimize")]
    [string]$Operation,
    
    [string]$BackupPath = "backups",
    [int]$RetentionDays = 7,
    [switch]$Force
)

# Error handling
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Import common functions
. "$PSScriptRoot\..\common\functions.ps1"

function Backup-System {
    Write-Log "Starting system backup..."
    
    # Create timestamp
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    
    # Create backup directory
    $backupDir = Join-Path $BackupPath $timestamp
    New-Item -ItemType Directory -Force -Path $backupDir
    
    try {
        # Backup database
        Write-Log "Backing up database..."
        pg_dump -h localhost -U postgres -F c -b -v -f "$backupDir\database.dump" ai_review_engine
        
        # Backup Redis data
        Write-Log "Backing up Redis data..."
        Copy-Item "C:\Program Files\Redis\dump.rdb" "$backupDir\redis.rdb"
        
        # Backup configuration
        Write-Log "Backing up configuration..."
        Copy-Item "config\*" "$backupDir\config" -Recurse
        
        # Create backup manifest
        @{
            timestamp = $timestamp
            components = @("database", "redis", "config")
            version = "1.0.0"
        } | ConvertTo-Json | Out-File "$backupDir\manifest.json"
        
        Write-Log "Backup completed successfully at $backupDir"
    }
    catch {
        Write-Log "Backup failed: $_" "ERROR"
        Remove-Item $backupDir -Recurse -Force
        exit 1
    }
}

function Clear-OldData {
    Write-Log "Starting cleanup operation..."
    
    # Clean old backups
    Get-ChildItem $BackupPath -Directory | Where-Object {
        $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays)
    } | ForEach-Object {
        Write-Log "Removing old backup: $($_.FullName)"
        Remove-Item $_.FullName -Recurse -Force
    }
    
    # Clean old logs
    Get-ChildItem "logs" -File | Where-Object {
        $_.LastWriteTime -lt (Get-Date).AddDays(-$RetentionDays)
    } | ForEach-Object {
        Write-Log "Removing old log: $($_.Name)"
        Remove-Item $_.FullName -Force
    }
    
    # Clean Redis cache
    Write-Log "Clearing Redis cache..."
    redis-cli FLUSHDB
    
    # Vacuum database
    Write-Log "Optimizing database..."
    psql -U postgres -d ai_review_engine -c "VACUUM FULL ANALYZE;"
    
    Write-Log "Cleanup completed successfully"
}

function Test-SystemHealth {
    Write-Log "Starting health check..."
    $healthy = $true
    
    # Check services
    $services = @(
        @{name="postgresql*"; display="PostgreSQL"},
        @{name="redis"; display="Redis"},
        @{name="AIReviewAPI"; display="API Service"},
        @{name="AIReviewWeb"; display="Web Service"},
        @{name="AIReviewDashboard"; display="Dashboard Service"},
        @{name="AIReviewWorker"; display="Worker Service"}
    )
    
    foreach ($service in $services) {
        $status = Get-Service $service.name -ErrorAction SilentlyContinue
        if ($status) {
            Write-Log "$($service.display) Status: $($status.Status)"
            if ($status.Status -ne "Running") {
                $healthy = $false
            }
        } else {
            Write-Log "$($service.display) not found" "WARNING"
            $healthy = $false
        }
    }
    
    # Check endpoints
    $endpoints = @(
        "http://localhost:8000/health",
        "http://localhost:5000/health",
        "http://localhost:8501/healthz"
    )
    
    foreach ($endpoint in $endpoints) {
        try {
            $response = Invoke-WebRequest -Uri $endpoint -UseBasicParsing
            Write-Log "$endpoint Status: $($response.StatusCode)"
            if ($response.StatusCode -ne 200) {
                $healthy = $false
            }
        }
        catch {
            Write-Log "$endpoint is not responding" "ERROR"
            $healthy = $false
        }
    }
    
    # Check database
    try {
        $dbResult = psql -U postgres -d ai_review_engine -c "\dt"
        Write-Log "Database connection successful"
    }
    catch {
        Write-Log "Database connection failed" "ERROR"
        $healthy = $false
    }
    
    # Check Redis
    try {
        $redisResult = redis-cli PING
        Write-Log "Redis connection successful"
    }
    catch {
        Write-Log "Redis connection failed" "ERROR"
        $healthy = $false
    }
    
    if ($healthy) {
        Write-Log "All systems are healthy!" "SUCCESS"
    } else {
        Write-Log "Some systems need attention" "WARNING"
    }
    
    return $healthy
}

function Get-SystemLogs {
    Write-Log "Collecting system logs..."
    
    # Create logs directory if it doesn't exist
    $logsDir = "logs\collected_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    New-Item -ItemType Directory -Force -Path $logsDir
    
    # Collect application logs
    Copy-Item "logs\*.log" $logsDir
    
    # Collect Windows Event Logs
    Get-EventLog -LogName Application -Source "AIReviewEngine" -Newest 100 |
        Export-Csv "$logsDir\windows_events.csv"
    
    # Collect service logs
    Get-Service "AIReview*" | Select-Object Name, Status, StartType |
        Export-Csv "$logsDir\services.csv"
    
    # Collect database logs
    pg_controldata | Out-File "$logsDir\postgresql_info.txt"
    
    # Create log archive
    Compress-Archive -Path $logsDir -DestinationPath "$logsDir.zip"
    Remove-Item $logsDir -Recurse -Force
    
    Write-Log "Logs collected and saved to $logsDir.zip"
}

function Optimize-System {
    Write-Log "Starting system optimization..."
    
    # Optimize database
    Write-Log "Optimizing database..."
    psql -U postgres -d ai_review_engine -c @"
    VACUUM FULL ANALYZE;
    REINDEX DATABASE ai_review_engine;
"@
    
    # Optimize Redis
    Write-Log "Optimizing Redis..."
    redis-cli BGREWRITEAOF
    
    # Clear temporary files
    Write-Log "Cleaning temporary files..."
    Get-ChildItem "temp" -Recurse | Remove-Item -Force
    
    # Optimize Windows services
    Write-Log "Optimizing services..."
    $services = Get-Service "AIReview*"
    foreach ($service in $services) {
        Restart-Service $service.Name -Force
    }
    
    Write-Log "System optimization completed"
}

# Main execution
try {
    switch ($Operation) {
        "backup" {
            Backup-System
        }
        "cleanup" {
            Clear-OldData
        }
        "healthcheck" {
            Test-SystemHealth
        }
        "logs" {
            Get-SystemLogs
        }
        "optimize" {
            Optimize-System
        }
    }
}
catch {
    Write-Log "Operation failed: $_" "ERROR"
    exit 1
}