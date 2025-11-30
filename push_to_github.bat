@echo off
echo ========================================
echo  Bird Song Generator - GitHub Push
echo ========================================
echo.

REM Check if git is initialized
if not exist .git (
    echo Initializing git repository...
    git init
    git add .
    git commit -m "Initial commit: Bird Song Generator with SimpleDiffusion"
    echo.
)

echo Step 1: Create GitHub Repository
echo --------------------------------
echo 1. Go to: https://github.com/new
echo 2. Repository name: bird-song-generator
echo 3. Make it PUBLIC
echo 4. DO NOT initialize with README
echo 5. Click "Create repository"
echo.
pause

echo.
echo Step 2: Enter Your GitHub Username
echo -----------------------------------
set /p USERNAME="GitHub username: "
echo.

echo Configuring remote...
git remote remove origin 2>nul
git remote add origin https://github.com/%USERNAME%/bird-song-generator.git
git branch -M main
echo Remote configured: https://github.com/%USERNAME%/bird-song-generator
echo.

echo Step 3: Push to GitHub
echo ----------------------
echo You'll need:
echo - Username: %USERNAME%
echo - Password: Use Personal Access Token (NOT your password!)
echo.
echo Get token at: https://github.com/settings/tokens
echo.
pause

echo Pushing to GitHub...
git push -u origin main

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo  SUCCESS! Repository pushed!
    echo ========================================
    echo.
    echo Your repository is live at:
    echo https://github.com/%USERNAME%/bird-song-generator
    echo.
    echo Share this link with your friends!
    echo.
) else (
    echo.
    echo ========================================
    echo  Push failed - Check your credentials
    echo ========================================
    echo.
    echo Make sure you're using a Personal Access Token
    echo Get one at: https://github.com/settings/tokens
    echo.
)

pause
