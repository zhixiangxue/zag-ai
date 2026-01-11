# ============================================================
# ZagPy Pre-Release Test Script
# ============================================================
# 
# This script validates zagpy installation and runs a complete
# E2E RAG pipeline test in a clean environment before releasing
# to PyPI. It ensures all dependencies work correctly.
#
# Usage:
#   .\scripts\pre_release.ps1
#
# What it does:
#   1. Creates a temporary virtual environment
#   2. Installs zagpy from the project directory
#   3. Runs complete E2E RAG pipeline test (no external services needed)
#   4. Cleans up the temporary environment
#
# Requirements:
#   - Ollama installed and running locally
#   - Model: ollama pull nomic-embed-text
#   - No other external services needed
#
# Exit codes:
#   0 - All tests passed
#   1 - Tests failed or error occurred
# ============================================================

$ErrorActionPreference = "Stop"

# Color output functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { Write-ColorOutput $args[0] "Green" }
function Write-Error { Write-ColorOutput $args[0] "Red" }
function Write-Info { Write-ColorOutput $args[0] "Cyan" }
function Write-Warning { Write-ColorOutput $args[0] "Yellow" }

# Get project root (parent of scripts directory)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$TempDir = Join-Path $env:TEMP "zagpy_install_test_$(Get-Date -Format 'yyyyMMdd_HHmmss')"

Write-Info "======================================================================"
Write-Info "  ZagPy Pre-Release Test"
Write-Info "====================================================================="
Write-Host ""
Write-Info "Project root: $ProjectRoot"
Write-Info "Test directory: $TempDir"
Write-Host ""

try {
    # Step 1: Create temporary virtual environment
    Write-Info "[1/5] Creating temporary virtual environment..."
    New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
    Push-Location $TempDir
    
    python -m venv test_env
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create virtual environment"
    }
    Write-Success "[OK] Virtual environment created"
    Write-Host ""
    
    # Step 2: Upgrade pip
    Write-Info "[2/5] Upgrading pip..."
    & ".\test_env\Scripts\python.exe" -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }
    Write-Success "[OK] Pip upgraded"
    Write-Host ""
    
    # Step 3: Install zagpy
    Write-Info "[3/5] Installing zagpy from: $ProjectRoot"
    Write-Warning "This may take a few minutes (downloading dependencies)..."
    & ".\test_env\Scripts\python.exe" -m pip install "$ProjectRoot"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install zagpy"
    }
    Write-Success "[OK] ZagPy installed"
    Write-Host ""
    
    # Step 4: Run E2E pre-release test
    Write-Info "[4/5] Running E2E pre-release test..."
    Write-Warning "Note: This test requires Ollama with jina-embeddings-v2-base-en model."
    Write-Warning "      Quick check: ollama list | findstr jina-embeddings-v2-base-en"
    Write-Host ""
    Write-Host "======================================================================"
    
    # Run test directly from project directory (avoid encoding issues with file copy)
    $TestFile = Join-Path $ProjectRoot "examples\e2e\pre_release_test.py"
    
    & ".\test_env\Scripts\python.exe" $TestFile
    $TestExitCode = $LASTEXITCODE
    
    Write-Host "======================================================================"
    Write-Host ""
    
    if ($TestExitCode -eq 0) {
        Write-Success "[OK] Pre-release test passed"
    } else {
        Write-Error "[FAIL] Pre-release test failed"
    }
    Write-Host ""
    
    # Step 5: Cleanup
    Write-Info "[5/5] Cleaning up temporary environment..."
    Pop-Location
    Start-Sleep -Seconds 1  # Give time for handles to release
    Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
    Write-Success "[OK] Cleanup completed"
    Write-Host ""
    
    # Final summary
    Write-Info "======================================================================"
    if ($TestExitCode -eq 0) {
        Write-Success "[SUCCESS] PRE-RELEASE TEST PASSED"
        Write-Host ""
        Write-Info "ZagPy is ready for release! Complete RAG pipeline validated:"
        Write-Host "  [OK] Document reading"
        Write-Host "  [OK] Text splitting & chunking"
        Write-Host "  [OK] Table parsing"
        Write-Host "  [OK] Embedding generation (local models)"
        Write-Host "  [OK] Vector indexing & storage"
        Write-Host "  [OK] Semantic retrieval"
        Write-Host "  [OK] Result postprocessing"
        Write-Host ""
        Write-Info "Next steps:"
        Write-Host "  1. Review the detailed test output above"
        Write-Host "  2. Build package: python -m build"
        Write-Host "  3. Publish to PyPI: twine upload dist/*"
    } else {
        Write-Error "[FAILED] PRE-RELEASE TEST FAILED"
        Write-Host ""
        Write-Warning "Please review the errors above and fix the issues."
        Write-Host ""
        Write-Info "Common issues:"
        Write-Host "  - Missing dependencies in pyproject.toml"
        Write-Host "  - Import errors in package code"
        Write-Host "  - Incorrect package structure"
        Write-Host "  - Component integration issues"
        Write-Host "  - Data processing errors"
    }
    Write-Info "======================================================================"
    Write-Host ""
    
    exit $TestExitCode
    
} catch {
    Write-Host ""
    Write-Error "[ERROR] Error occurred: $_"
    Write-Host ""
    
    # Cleanup on error
    Pop-Location -ErrorAction SilentlyContinue
    if (Test-Path $TempDir) {
        Remove-Item -Recurse -Force $TempDir -ErrorAction SilentlyContinue
    }
    
    exit 1
}
