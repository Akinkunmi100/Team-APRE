# 📱 WhatsApp Integration Guide

## Overview

The WhatsApp integration allows users to submit and manage reviews directly through WhatsApp. This integration uses the WhatsApp Business API to handle messages and provide a conversational interface for the AI Review Engine.

## Features

1. **Review Submission**
   - Submit reviews via text messages
   - Include ratings and comments
   - Receive confirmation messages

2. **System Status**
   - Check system health
   - View API status
   - Monitor service availability

3. **Help and Documentation**
   - Access command list
   - View usage examples
   - Get instant help

## Setup Instructions

### 1. Prerequisites

- WhatsApp Business API account
- Meta Developer account
- Valid SSL certificate for webhook endpoint
- Public IP address or domain name

### 2. Installation

```powershell
# Clone repository (if not already done)
cd C:\Users\OLANREWAJU BDE\Desktop\ai-review-engine_updated
cd whatsapp

# Run installation script
.\install.ps1

# For production installation
.\install.ps1 -Production
```

### 3. Configuration

1. Copy the example environment file:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Update the environment variables in `.env`:
   ```plaintext
   WHATSAPP_API_URL=https://graph.facebook.com/v17.0
   WHATSAPP_PHONE_ID=your_phone_number_id
   WHATSAPP_ACCESS_TOKEN=your_access_token
   WHATSAPP_VERIFY_TOKEN=your_webhook_verify_token
   WHATSAPP_APP_SECRET=your_app_secret
   ```

3. Configure webhook in Meta Developer Console:
   - URL: `https://your-domain.com/webhook`
   - Verify Token: Same as `WHATSAPP_VERIFY_TOKEN`

## Usage Guide

### Available Commands

1. Submit a Review:
   ```plaintext
   review <product_id> <rating> <text>
   
   Example:
   review 12345 5 Great product! Very satisfied with the quality.
   ```

2. Check Status:
   ```plaintext
   status
   ```

3. Get Help:
   ```plaintext
   help
   ```

### Message Flow

1. **Review Submission**:
   ```mermaid
   sequenceDiagram
       User->>WhatsApp: review 12345 5 Great product!
       WhatsApp->>Service: Forward message
       Service->>API: Submit review
       API->>Service: Confirm submission
       Service->>WhatsApp: Send confirmation
       WhatsApp->>User: Review submitted successfully!
   ```

2. **Status Check**:
   ```mermaid
   sequenceDiagram
       User->>WhatsApp: status
       WhatsApp->>Service: Forward command
       Service->>API: Check health
       API->>Service: Return status
       Service->>WhatsApp: Send status
       WhatsApp->>User: System Status: Online
   ```

## Development Guide

### Project Structure

```plaintext
whatsapp/
├── main.py               # FastAPI application
├── whatsapp_service.py   # WhatsApp integration logic
├── requirements.txt      # Production dependencies
├── requirements-dev.txt  # Development dependencies
├── install.ps1          # Installation script
└── .env.example         # Environment template
```

### Running Tests

```powershell
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

### Development Server

```powershell
# Start development server
uvicorn main:app --reload --port 8002
```

## Security Considerations

1. **API Security**
   - Use HTTPS for all endpoints
   - Validate webhook signatures
   - Implement rate limiting

2. **Data Privacy**
   - Store phone numbers securely
   - Encrypt sensitive data
   - Implement data retention policies

3. **Access Control**
   - Validate WhatsApp Business API tokens
   - Implement user authentication
   - Monitor access logs

## Monitoring and Logging

### Log Files

```plaintext
logs/
├── app.log         # Application logs
├── access.log      # Access logs
├── error.log       # Error logs
└── webhook.log     # Webhook logs
```

### Monitoring Endpoints

1. Health Check:
   ```http
   GET /health
   ```

2. Metrics:
   ```http
   GET /metrics
   ```

## Troubleshooting

### Common Issues

1. **Webhook Verification Failed**
   - Check verify token matches
   - Ensure HTTPS is configured
   - Verify webhook URL is accessible

2. **Message Sending Failed**
   - Verify access token is valid
   - Check rate limits
   - Ensure phone number is verified

3. **API Connection Issues**
   - Check main API is running
   - Verify network connectivity
   - Check API credentials

### Debug Mode

Enable debug mode in `.env`:
```plaintext
DEBUG=true
LOG_LEVEL=DEBUG
```

## Best Practices

1. **Message Handling**
   - Implement message queuing
   - Handle message retries
   - Validate message format

2. **Error Handling**
   - Implement graceful fallbacks
   - Log all errors
   - Send user-friendly error messages

3. **Performance**
   - Cache frequently used data
   - Implement message batching
   - Monitor resource usage

## Integration Testing

### Test Environment

1. Setup test environment:
   ```powershell
   # Create test environment
   python -m venv test_env
   .\test_env\Scripts\Activate
   pip install -r requirements-dev.txt
   ```

2. Run integration tests:
   ```powershell
   pytest tests/integration/
   ```

### Mock WhatsApp API

Use the provided mock server for testing:
```powershell
# Start mock server
python tests/mock_whatsapp_api.py
```

## Deployment

### Production Deployment

1. Set up SSL certificate:
   ```powershell
   # Generate SSL certificate
   .\security\ssl\generate_cert.ps1 -Domain your-domain.com
   ```

2. Configure NGINX:
   ```nginx
   server {
       listen 443 ssl;
       server_name your-domain.com;
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       
       location / {
           proxy_pass http://localhost:8002;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

3. Start service:
   ```powershell
   # As Windows service
   Start-Service WhatsAppIntegration
   ```

### Docker Deployment

```powershell
# Build Docker image
docker build -t ai-review-whatsapp .

# Run container
docker run -d -p 8002:8002 ai-review-whatsapp
```

## Updates and Maintenance

### Update Procedure

1. Stop service:
   ```powershell
   Stop-Service WhatsAppIntegration
   ```

2. Update code:
   ```powershell
   git pull origin main
   ```

3. Update dependencies:
   ```powershell
   pip install -r requirements.txt --upgrade
   ```

4. Start service:
   ```powershell
   Start-Service WhatsAppIntegration
   ```

### Backup and Recovery

1. Backup configuration:
   ```powershell
   Copy-Item .env .env.backup
   ```

2. Restore configuration:
   ```powershell
   Copy-Item .env.backup .env
   ```