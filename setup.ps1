# ARIA — Windows setup script (PowerShell)
# Installs Python deps + Ollama models. Safe to re-run.
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "── ARIA setup (Windows) ──────────────────────" -ForegroundColor Cyan

# 1. Python virtualenv
if (-not (Test-Path ".venv")) {
    Write-Host "▶ Creating virtualenv" -ForegroundColor Yellow
    python -m venv .venv
}

# Activate
& .\.venv\Scripts\Activate.ps1

# 2. Dependencies + install the package
Write-Host "▶ Installing Python dependencies" -ForegroundColor Yellow
python -m pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet -e .

# 3. Ollama (if missing)
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    Write-Host "▶ Ollama not found. Installing..." -ForegroundColor Yellow
    Write-Host "  Downloading Ollama installer..."
    $installer = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $installer
    Write-Host "  Running installer (may require admin)..."
    Start-Process -FilePath $installer -Wait
    Write-Host "  ✓ Ollama installed. You may need to restart your terminal."
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
} else {
    Write-Host "  Ollama found: $($ollama.Source)" -ForegroundColor Green
}

# Ensure Ollama server is running
$ollamaProcess = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProcess) {
    Write-Host "▶ Starting Ollama server..." -ForegroundColor Yellow
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
}

# 4. Pull models (skipped if already present)
function Pull-IfMissing($model) {
    $list = ollama list 2>$null
    if ($list -match $model) {
        Write-Host "  $model already present" -ForegroundColor Green
    } else {
        Write-Host "▶ Pulling $model (~1-2 GB download)" -ForegroundColor Yellow
        ollama pull $model
    }
}

Pull-IfMissing "granite4.1:3b"      # main model (chat + code)
Pull-IfMissing "hermes3:3b"         # fallback provider (auto-failover)
Pull-IfMissing "nomic-embed-text"   # embeddings for semantic memory

# 5. .env
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "▶ Created .env (defaults are fine)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "✓ ARIA ready. Start it with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1; aria" -ForegroundColor White
Write-Host "  (start Ollama first if it isn't running: ollama serve)" -ForegroundColor Gray
