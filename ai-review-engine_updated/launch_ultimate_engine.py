#!/usr/bin/env python3
"""
Launch Ultimate AI Phone Review Engine
Quick start script for the professional web application
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Ultimate AI Phone Review Engine - Launch Script")
    print("=" * 55)
    
    # Check if virtual environment exists
    venv_path = Path("main_venv")
    if not venv_path.exists():
        print("❌ Virtual environment 'main_venv' not found!")
        print("Please run the setup first or activate your environment manually.")
        sys.exit(1)
    
    # Check for CSV data file
    csv_file = Path("final_dataset_streamlined_clean.csv")
    if not csv_file.exists():
        print("⚠️  Warning: CSV data file not found. The app will work but with limited data.")
    else:
        print(f"✅ Data file found: {len(open(csv_file).readlines())} lines of review data")
    
    print("\n🔧 Starting Ultimate AI Phone Review Engine...")
    print("📍 Application will be available at: http://localhost:5000")
    print("\n🎮 Demo Accounts:")
    print("   🆓 Free User:       demo_user / demo123")
    print("   🏢 Business User:   business_user / business123")
    print("   🚀 Enterprise User: enterprise_user / enterprise123")
    print("\n" + "=" * 55)
    
    # Determine the correct Python executable
    if sys.platform == "win32":
        python_exe = venv_path / "Scripts" / "python.exe"
        if not python_exe.exists():
            python_exe = "python"
    else:
        python_exe = venv_path / "bin" / "python"
        if not python_exe.exists():
            python_exe = "python3"
    
    # Launch the app
    try:
        subprocess.run([str(python_exe), "ultimate_web_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running application: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Activate your virtual environment:")
        print("   Windows: main_venv\\Scripts\\activate")
        print("   Linux/Mac: source main_venv/bin/activate")
        print("3. Install dependencies: pip install -r requirements_ultimate.txt")
        print("4. Run manually: python ultimate_web_app.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()