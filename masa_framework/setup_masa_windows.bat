@echo off
REM MASA Framework Setup Script for Windows
REM This script sets up the MASA framework in your specified directory

echo ========================================
echo MASA Framework Setup for Windows
echo ========================================

REM Set the target directory
set TARGET_DIR=C:\Users\user\Documents\Neural Network Trading\masa_framework

echo.
echo Target directory: %TARGET_DIR%

REM Create the target directory if it doesn't exist
if not exist "C:\Users\user\Documents\Neural Network Trading" (
    echo Creating Neural Network Trading directory...
    mkdir "C:\Users\user\Documents\Neural Network Trading"
)

if not exist "%TARGET_DIR%" (
    echo Creating masa_framework directory...
    mkdir "%TARGET_DIR%"
)

echo.
echo Directory structure created successfully!

echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo 1. Extract the masa_framework.tar.gz file to:
echo    %TARGET_DIR%
echo.
echo 2. Install Python dependencies:
echo    cd "%TARGET_DIR%"
echo    pip install -r requirements.txt
echo.
echo 3. Test the installation:
echo    python example_usage.py
echo.
echo 4. Explore the demo notebook:
echo    jupyter notebook masa_demo.ipynb
echo.
echo ========================================
echo Manual Setup Instructions:
echo ========================================
echo.
echo If you prefer manual setup:
echo.
echo 1. Download the masa_framework.tar.gz file
echo 2. Extract it using 7-Zip, WinRAR, or Windows built-in extraction
echo 3. Copy the extracted masa_framework folder to:
echo    C:\Users\user\Documents\Neural Network Trading\
echo 4. Open Command Prompt or PowerShell in that directory
echo 5. Run: pip install -r requirements.txt
echo 6. Test with: python example_usage.py
echo.

pause