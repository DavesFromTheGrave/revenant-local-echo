# Revenant Echo — Launcher
#
# Checks that Ollama is reachable, then launches V on Python 3.11.
# Use this script instead of calling main.py directly so the Ollama
# health check happens before V tries to warm the model.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run.ps1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host "Revenant Echo — Launcher" -ForegroundColor Cyan
Write-Host ""

# ─── Ollama health check ─────────────────────────────────────────────────
$backendType = Select-String -Path "config\config.yaml" -Pattern '^\s*type:\s*"?(\w+)' |
    ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1

if ($backendType -eq "ollama") {
    Write-Host "Checking Ollama backend…" -ForegroundColor Green
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        if ($response.StatusCode -eq 200) {
            Write-Host "  Ollama is running" -ForegroundColor Green
        }
    } catch {
        Write-Host "  Ollama not responding at http://localhost:11434" -ForegroundColor Yellow
        Write-Host "  Start it in another terminal:  ollama serve" -ForegroundColor Yellow
        Write-Host "  Or just start the Ollama desktop app." -ForegroundColor Gray
        Write-Host "  Proceeding anyway (V will fail to warm the model)…" -ForegroundColor Gray
    }
    Write-Host ""
}

# ─── Launch V ────────────────────────────────────────────────────────────
Write-Host "Starting V… (Ctrl+C to stop)" -ForegroundColor Green
Write-Host ""

& py -3.11 src\main.py

Write-Host ""
Write-Host "V stopped" -ForegroundColor Cyan
