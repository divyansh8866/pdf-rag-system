#!/usr/bin/env python3
"""
Setup script for Mini RAG System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_directories():
    """Create necessary directories."""
    directories = ["data", "store"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_pdfs():
    """Check if there are PDF files in the data directory."""
    pdf_files = list(Path("data").glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in data/ directory")
        print("   Please add some PDF files to test the system")
        return False
    else:
        print(f"✅ Found {len(pdf_files)} PDF file(s) in data/ directory")
        return True

def main():
    """Main setup function."""
    print("🚀 Setting up Mini RAG System...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check for virtual environment
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Virtual environment not detected")
        print("   It's recommended to create a virtual environment:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check for PDFs
    check_pdfs()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Add PDF files to the data/ directory")
    print("2. Run: python app/ingest.py")
    print("3. Start the API: uvicorn app.api:app --reload")
    print("4. Visit: http://127.0.0.1:8000/docs")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 