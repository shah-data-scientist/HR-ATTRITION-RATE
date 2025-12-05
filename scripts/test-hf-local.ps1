# Test Hugging Face Spaces deployment locally
# This script simulates the HF Spaces environment with in-memory database

Write-Host "=== Testing HF Spaces Deployment Locally ===" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "[1/6] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "  ✓ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check if container is already running
$existingContainer = docker ps -a -q -f name=test-hf-space
if ($existingContainer) {
    Write-Host "[2/6] Removing existing test container..." -ForegroundColor Yellow
    docker stop test-hf-space 2>$null
    docker rm test-hf-space 2>$null
    Write-Host "  ✓ Cleaned up existing container" -ForegroundColor Green
}

# Build the image
Write-Host "[3/6] Building HF Spaces container..." -ForegroundColor Yellow
Write-Host "  (This may take 5-10 minutes on first build)" -ForegroundColor Gray
$buildResult = docker build -f docker/Dockerfile.huggingface -t hr-attrition-hf:test . 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Build failed" -ForegroundColor Red
    Write-Host $buildResult
    exit 1
}
Write-Host "  ✓ Container built successfully" -ForegroundColor Green

# Run the container with in-memory database
Write-Host "[4/6] Starting container..." -ForegroundColor Yellow
docker run -d -p 7860:7860 -p 8001:8001 `
  -e DATABASE_URL="sqlite:///:memory:" `
  -e DISABLE_DB="0" `
  -e UI_ADMIN_USERNAME="admin" `
  -e UI_ADMIN_PASSWORD="TestAdmin123!" `
  -e UI_USER_USERNAME="analyst" `
  -e UI_USER_PASSWORD="TestAnalyst456!" `
  --name test-hf-space `
  hr-attrition-hf:test

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ✗ Failed to start container" -ForegroundColor Red
    exit 1
}
Write-Host "  ✓ Container started" -ForegroundColor Green

# Wait for services to be ready
Write-Host "[5/6] Waiting for services to start..." -ForegroundColor Yellow
Write-Host "  (This takes about 30-60 seconds)" -ForegroundColor Gray

$maxAttempts = 30
$attempt = 0
$ready = $false

while ($attempt -lt $maxAttempts -and -not $ready) {
    Start-Sleep -Seconds 2
    $attempt++

    try {
        # Check if both API and UI are responding
        $apiResponse = Invoke-WebRequest -Uri "http://localhost:8001/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
        $uiResponse = Invoke-WebRequest -Uri "http://localhost:7860" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue

        if ($apiResponse.StatusCode -eq 200 -and $uiResponse.StatusCode -eq 200) {
            $ready = $true
        }
    } catch {
        Write-Host "  ." -NoNewline -ForegroundColor Gray
    }
}

if (-not $ready) {
    Write-Host ""
    Write-Host "  ⚠ Services did not start in time. Checking logs..." -ForegroundColor Yellow
    Write-Host ""
    docker logs test-hf-space --tail 50
    Write-Host ""
    Write-Host "  Container is still running. You can check logs with:" -ForegroundColor Yellow
    Write-Host "  docker logs -f test-hf-space" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "  ✓ Services are ready!" -ForegroundColor Green
}

# Show container logs to verify database initialization
Write-Host ""
Write-Host "[6/6] Checking database initialization..." -ForegroundColor Yellow
$logs = docker logs test-hf-space 2>&1 | Select-String -Pattern "Creating default users|Successfully created"
if ($logs) {
    Write-Host "  ✓ Database initialization logs:" -ForegroundColor Green
    $logs | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
} else {
    Write-Host "  ⚠ Database logs not found yet. Check full logs:" -ForegroundColor Yellow
    Write-Host "    docker logs test-hf-space" -ForegroundColor Gray
}

# Summary
Write-Host ""
Write-Host "=== Test Environment Ready ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Access the application:" -ForegroundColor White
Write-Host "  🌐 UI:  http://localhost:7860" -ForegroundColor Green
Write-Host "  🔧 API: http://localhost:8001/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Login credentials:" -ForegroundColor White
Write-Host "  👤 Admin:" -ForegroundColor Cyan
Write-Host "     Username: admin" -ForegroundColor Gray
Write-Host "     Password: TestAdmin123!" -ForegroundColor Gray
Write-Host "  👤 Analyst:" -ForegroundColor Cyan
Write-Host "     Username: analyst" -ForegroundColor Gray
Write-Host "     Password: TestAnalyst456!" -ForegroundColor Gray
Write-Host ""
Write-Host "Useful commands:" -ForegroundColor White
Write-Host "  View logs:      docker logs -f test-hf-space" -ForegroundColor Gray
Write-Host "  Stop container: docker stop test-hf-space" -ForegroundColor Gray
Write-Host "  Remove:         docker rm test-hf-space" -ForegroundColor Gray
Write-Host ""
Write-Host "⚠ Note: Using in-memory database - data resets on restart" -ForegroundColor Yellow
Write-Host ""

# Open browser
$openBrowser = Read-Host "Open browser now? (Y/n)"
if ($openBrowser -ne 'n' -and $openBrowser -ne 'N') {
    Start-Process "http://localhost:7860"
}
