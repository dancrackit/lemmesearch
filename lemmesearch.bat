@echo off
setlocal enabledelayedexpansion

echo [1/4] Checking virtual environment...
if not exist .venv (
    echo Creating virtual environment in .venv...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo Failed to create virtual environment. Ensure Python is installed and in your PATH.
        pause
        exit /b 1
    )
)

echo [2/4] Checking and migrating data from .venv\app if needed...
:: Migrate existing credentials from .venv\app back to root if they exist there
if exist .venv\app\credential (
    echo Migrating credentials from .venv\app to codebase root...
    if not exist credential mkdir credential
    xcopy /S /Y /I /E .venv\app\credential credential >nul
)
:: Migrate database from .venv\app back to root if they exist there
if exist .venv\app\chroma_db (
    echo Migrating database from .venv\app to codebase root...
    if not exist chroma_db mkdir chroma_db
    xcopy /S /Y /I /E .venv\app\chroma_db chroma_db >nul
)
:: Migrate scratch files from .venv\app back to root if they exist there
if exist .venv\app\scratch (
    echo Migrating scratch files from .venv\app to codebase root...
    if not exist scratch mkdir scratch
    xcopy /S /Y /I /E .venv\app\scratch scratch >nul
)
:: Migrate chat history from .venv\app\.venv\chat_history back to credential\chat_history if exists
if exist .venv\app\.venv\chat_history (
    echo Migrating chat history from .venv\app to credential\chat_history...
    if not exist credential\chat_history mkdir credential\chat_history
    xcopy /S /Y /I /E .venv\app\.venv\chat_history credential\chat_history >nul
)
:: Migrate chat history from .venv\chat_history back to credential\chat_history if exists
if exist .venv\chat_history (
    echo Migrating chat history from .venv\chat_history to credential\chat_history...
    if not exist credential\chat_history mkdir credential\chat_history
    xcopy /S /Y /I /E .venv\chat_history credential\chat_history >nul
    rmdir /S /Q .venv\chat_history
)

:: Remove the duplicate .venv\app directory if it exists
if exist .venv\app (
    echo Cleaning up duplicate app folder in .venv...
    rmdir /S /Q .venv\app
)

echo [3/4] Installing dependencies...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo Failed to install dependencies.
    pause
    exit /b 1
)

echo [4/4] Starting backend server on http://localhost:7777...
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 7777 --reload
