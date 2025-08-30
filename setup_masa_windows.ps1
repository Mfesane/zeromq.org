# MASA Framework Setup Script for Windows PowerShell
# This script sets up the MASA framework in your specified directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "MASA Framework Setup for Windows" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Set the target directory
$TargetDir = "C:\Users\user\Documents\Neural Network Trading\masa_framework"
$BaseDir = "C:\Users\user\Documents\Neural Network Trading"

Write-Host ""
Write-Host "Target directory: $TargetDir" -ForegroundColor Yellow

# Create the target directory if it doesn't exist
if (!(Test-Path $BaseDir)) {
    Write-Host "Creating Neural Network Trading directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path $BaseDir -Force
}

if (!(Test-Path $TargetDir)) {
    Write-Host "Creating masa_framework directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path $TargetDir -Force
}

Write-Host ""
Write-Host "Directory structure created successfully!" -ForegroundColor Green

# Check if Python is installed
Write-Host ""
Write-Host "Checking Python installation..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Host "Python found: $pythonVersion" -ForegroundColor Green
    } else {
        Write-Host "Python not found in PATH. Please install Python 3.8+" -ForegroundColor Red
    }
} catch {
    Write-Host "Python not found. Please install Python 3.8+" -ForegroundColor Red
}

# Check if pip is available
try {
    $pipVersion = pip --version 2>$null
    if ($pipVersion) {
        Write-Host "Pip found: $pipVersion" -ForegroundColor Green
    }
} catch {
    Write-Host "Pip not found. Please ensure pip is installed" -ForegroundColor Red
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Instructions:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "1. Extract masa_framework.tar.gz to:" -ForegroundColor White
Write-Host "   $TargetDir" -ForegroundColor Yellow

Write-Host ""
Write-Host "2. Open PowerShell/Command Prompt and navigate to:" -ForegroundColor White
Write-Host "   cd `"$TargetDir`"" -ForegroundColor Yellow

Write-Host ""
Write-Host "3. Install Python dependencies:" -ForegroundColor White
Write-Host "   pip install -r requirements.txt" -ForegroundColor Yellow

Write-Host ""
Write-Host "4. Test the installation:" -ForegroundColor White
Write-Host "   python example_usage.py" -ForegroundColor Yellow

Write-Host ""
Write-Host "5. Explore the demo notebook:" -ForegroundColor White
Write-Host "   jupyter notebook masa_demo.ipynb" -ForegroundColor Yellow

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Alternative: Manual Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "If automatic extraction doesn't work:" -ForegroundColor White
Write-Host "1. Download masa_framework.tar.gz" -ForegroundColor Gray
Write-Host "2. Use 7-Zip, WinRAR, or Windows extraction" -ForegroundColor Gray
Write-Host "3. Copy extracted folder to the target directory" -ForegroundColor Gray
Write-Host "4. Follow steps 2-5 above" -ForegroundColor Gray

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Troubleshooting:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "If you encounter issues:" -ForegroundColor White
Write-Host "• Ensure Python 3.8+ is installed" -ForegroundColor Gray
Write-Host "• Run PowerShell as Administrator if needed" -ForegroundColor Gray
Write-Host "• Use 'python3' instead of 'python' if needed" -ForegroundColor Gray
Write-Host "• Create virtual environment: python -m venv masa_env" -ForegroundColor Gray
Write-Host "• Activate venv: masa_env\Scripts\activate" -ForegroundColor Gray

Write-Host ""
Write-Host "Setup script completed!" -ForegroundColor Green
Write-Host "Press any key to continue..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")