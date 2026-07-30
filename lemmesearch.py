import os
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.resolve()
    venv_dir = root_dir / ".venv"
    
    # 1. Virtual Environment Setup
    print("[1/4] Checking virtual environment...")
    if not venv_dir.exists():
        print(f"Creating virtual environment in {venv_dir}...")
        try:
            import venv
            # Create venv with pip enabled
            venv.create(venv_dir, with_pip=True)
        except Exception as e:
            print(f"Failed to create virtual environment: {e}")
            print("Please ensure Python 3 is installed with the 'venv' package available.")
            sys.exit(1)
            
    # Determine the python/pip paths based on OS platform
    if os.name == "nt":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    if not venv_python.exists():
        print(f"Error: Virtual environment python executable not found at {venv_python}")
        sys.exit(1)

    # 2. Data Migration from legacy .venv/app paths if necessary
    print("[2/4] Checking and migrating data from legacy paths if needed...")
    legacy_app_dir = venv_dir / "app"
    if legacy_app_dir.exists():
        # Migrate credentials
        legacy_cred = legacy_app_dir / "credential"
        if legacy_cred.exists():
            print("Migrating credentials to codebase root...")
            shutil.copytree(legacy_cred, root_dir / "credential", dirs_exist_ok=True)
            
        # Migrate database
        legacy_db = legacy_app_dir / "chroma_db"
        if legacy_db.exists():
            print("Migrating database to codebase root...")
            shutil.copytree(legacy_db, root_dir / "chroma_db", dirs_exist_ok=True)
            
        # Migrate scratch files
        legacy_scratch = legacy_app_dir / "scratch"
        if legacy_scratch.exists():
            print("Migrating scratch files to codebase root...")
            shutil.copytree(legacy_scratch, root_dir / "scratch", dirs_exist_ok=True)
            
        # Migrate chat history from .venv/app/.venv/chat_history
        legacy_history = legacy_app_dir / ".venv" / "chat_history"
        if legacy_history.exists():
            print("Migrating chat history to codebase root...")
            shutil.copytree(legacy_history, root_dir / "credential" / "chat_history", dirs_exist_ok=True)
            
    # Migrate chat history from .venv/chat_history
    legacy_history_venv = venv_dir / "chat_history"
    if legacy_history_venv.exists():
        print("Migrating chat history from .venv/chat_history...")
        shutil.copytree(legacy_history_venv, root_dir / "credential" / "chat_history", dirs_exist_ok=True)
        try:
            shutil.rmtree(legacy_history_venv)
        except Exception:
            pass

    # Clean up legacy app directory
    if legacy_app_dir.exists():
        print("Cleaning up duplicate app folder in .venv...")
        try:
            shutil.rmtree(legacy_app_dir)
        except Exception as e:
            print(f"Warning: Could not remove {legacy_app_dir}: {e}")

    # 3. Installing dependencies
    print("[3/4] Installing and upgrading dependencies...")
    # Upgrade pip inside venv
    try:
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to upgrade pip: {e}")

    # Install requirements
    requirements_path = root_dir / "requirements.txt"
    if requirements_path.exists():
        try:
            subprocess.run([str(venv_python), "-m", "pip", "install", "-r", str(requirements_path)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies from requirements.txt: {e}")
            sys.exit(1)
    else:
        # Fallback to installing from pyproject.toml
        try:
            print("No requirements.txt found. Installing from pyproject.toml...")
            subprocess.run([str(venv_python), "-m", "pip", "install", "-e", "."], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to install dependencies from pyproject.toml: {e}")
            sys.exit(1)

    # 4. Launching the Backend Server
    print("[4/4] Starting backend server on http://localhost:7777...")
    try:
        # Run uvicorn without --reload flag
        subprocess.run([
            str(venv_python), "-m", "uvicorn", 
            "backend.main:app", 
            "--host", "127.0.0.1", 
            "--port", "7777"
        ], check=True)
    except KeyboardInterrupt:
        print("\nStopping lemmesearch server.")
    except subprocess.CalledProcessError as e:
        print(f"Server exited with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
