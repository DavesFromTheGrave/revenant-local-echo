# Revenant Echo — Installation Script
#
# Sets up Python 3.11 dependencies for the voice assistant on Windows.
# Does NOT create a venv — uses your system Python 3.11 directly via the
# `py -3.11` launcher. Install Python 3.11 from python.org first if you
# don't have it.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Revenant Echo — Installation" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""

# ─── 1. Python 3.11 ───────────────────────────────────────────────────────
Write-Host "Step 1: Checking Python 3.11…" -ForegroundColor Green
try {
    $pyVersion = & py -3.11 --version
    Write-Host "  Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python 3.11 not found via the 'py' launcher" -ForegroundColor Red
    Write-Host "  Install Python 3.11 from python.org and re-run." -ForegroundColor Yellow
    exit 1
}
Write-Host ""

# ─── 2. CUDA ──────────────────────────────────────────────────────────────
Write-Host "Step 2: Checking NVIDIA driver / CUDA…" -ForegroundColor Green
$nvidiaCmd = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaCmd) {
    $cudaVersion = & nvidia-smi --query-gpu=driver_version --format=csv,noheader
    Write-Host "  NVIDIA Driver: $cudaVersion" -ForegroundColor Green
    Write-Host "  GPU acceleration available." -ForegroundColor Green
} else {
    Write-Host "  nvidia-smi not found — CPU-only mode (slow)." -ForegroundColor Yellow
}
Write-Host ""

# ─── 3. Core dependencies ─────────────────────────────────────────────────
Write-Host "Step 3: Installing core dependencies…" -ForegroundColor Green
Write-Host "  (may take 5–15 minutes on first run)" -ForegroundColor Gray
& py -3.11 -m pip install --upgrade pip
& py -3.11 -m pip install faster-whisper "openwakeword>=0.6.0" pyaudio numpy requests pyyaml python-dotenv
if ($LASTEXITCODE -ne 0) { Write-Host "  pip install failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Step 3b: Installing CUDA-enabled torch (cu121)…" -ForegroundColor Green
& py -3.11 -m pip install torch --index-url https://download.pytorch.org/whl/cu121
if ($LASTEXITCODE -ne 0) { Write-Host "  torch install failed" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "Step 3c: Installing Kokoro TTS (no-deps, then runtime deps)…" -ForegroundColor Green
# kokoro hard-pins numpy==1.26.4 which is overly conservative; --no-deps
# skips that and we install kokoro's actual runtime deps explicitly.
& py -3.11 -m pip install --no-deps kokoro==0.9.4
& py -3.11 -m pip install loguru huggingface_hub "misaki[en]" munch espeakng-loader transformers
if ($LASTEXITCODE -ne 0) { Write-Host "  kokoro install failed" -ForegroundColor Red; exit 1 }
Write-Host ""

# ─── 4. Wake word model ───────────────────────────────────────────────────
Write-Host "Step 4: Fetching wake-word model (Hey Friday)…" -ForegroundColor Green
$modelsDir = Join-Path $projectRoot "models"
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir | Out-Null
}
$wakeModel = Join-Path $modelsDir "hey_friday.onnx"
if (Test-Path $wakeModel) {
    Write-Host "  Already present: $wakeModel" -ForegroundColor Green
} else {
    $url = "https://github.com/fwartner/home-assistant-wakewords-collection/raw/main/en/hey_friday/hey_Friday%21.onnx"
    Invoke-WebRequest -Uri $url -OutFile $wakeModel
    Write-Host "  Downloaded to: $wakeModel" -ForegroundColor Green
}
Write-Host ""

# ─── 5. OpenWakeWord support models ───────────────────────────────────────
Write-Host "Step 5: Downloading OpenWakeWord support models…" -ForegroundColor Green
& py -3.11 -c "import openwakeword.utils; openwakeword.utils.download_models()"
Write-Host ""

# ─── Done ─────────────────────────────────────────────────────────────────
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host "Installation complete." -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Start Ollama (desktop app or:  ollama serve)" -ForegroundColor Gray
Write-Host "  2. Pull your model:  ollama pull <your-model-name>" -ForegroundColor Gray
Write-Host "  3. Edit config\config.yaml to point at your model and mic" -ForegroundColor Gray
Write-Host "  4. Launch V:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1" -ForegroundColor Gray
Write-Host ""
