# Generate SSL Certificate Script
param(
    [string]$Domain = "localhost",
    [string]$OutputPath = ".\certs",
    [int]$ValidDays = 365
)

# Create output directory if it doesn't exist
New-Item -ItemType Directory -Force -Path $OutputPath

# Generate OpenSSL configuration
$opensslConfig = @"
[req]
default_bits = 2048
prompt = no
default_md = sha256
x509_extensions = v3_req
distinguished_name = dn

[dn]
C = US
ST = State
L = City
O = Organization
OU = Unit
CN = $Domain

[v3_req]
subjectAltName = @alt_names
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment

[alt_names]
DNS.1 = $Domain
DNS.2 = *.$Domain
DNS.3 = localhost
IP.1 = 127.0.0.1
"@

# Save OpenSSL configuration
$opensslConfig | Out-File -FilePath "$OutputPath\openssl.cnf" -Encoding ascii

# Generate private key and certificate
Write-Host "Generating SSL certificate for $Domain..."
openssl req `
    -x509 `
    -nodes `
    -days $ValidDays `
    -newkey rsa:2048 `
    -keyout "$OutputPath\private.key" `
    -out "$OutputPath\certificate.crt" `
    -config "$OutputPath\openssl.cnf"

# Generate PFX file for Windows
Write-Host "Generating PFX file..."
openssl pkcs12 `
    -export `
    -out "$OutputPath\certificate.pfx" `
    -inkey "$OutputPath\private.key" `
    -in "$OutputPath\certificate.crt" `
    -passout pass:

# Set permissions
$acl = Get-Acl "$OutputPath\private.key"
$acl.SetAccessRuleProtection($true, $false)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("SYSTEM","FullControl","Allow")
$acl.AddAccessRule($rule)
$rule = New-Object System.Security.AccessControl.FileSystemAccessRule("Administrators","FullControl","Allow")
$acl.AddAccessRule($rule)
Set-Acl "$OutputPath\private.key" $acl

Write-Host "`nSSL Certificate generated successfully!"
Write-Host "Files created:"
Write-Host "- $OutputPath\private.key"
Write-Host "- $OutputPath\certificate.crt"
Write-Host "- $OutputPath\certificate.pfx"