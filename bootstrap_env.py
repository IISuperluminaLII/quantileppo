
import os
import sys
import subprocess
import platform
from pathlib import Path

VENV_DIR = Path(".venv")
IS_WINDOWS = platform.system().lower().startswith("win")

def venv_python():
    if IS_WINDOWS:
        return VENV_DIR / "Scripts" / "python.exe"
    else:
        return VENV_DIR / "bin" / "python"

def ensure_venv():
    if not VENV_DIR.exists():
        print("Creating virtual environment in .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        print("Using existing .venv")

def pip_install(args):
    py = str(venv_python())
    cmd = [py, "-m", "pip"] + args
    subprocess.check_call(cmd)

def main():
    ensure_venv()
    # Upgrade pip & wheel/setuptools
    print("Upgrading pip, setuptools, wheel ...")
    pip_install(["install", "--upgrade", "pip", "setuptools", "wheel"])

    req = Path("requirements.txt")
    if req.exists():
        print("Installing requirements.txt ...")
        pip_install(["install", "-r", str(req)])
    else:
        print("requirements.txt not found, skipping dependency install.")

    # Final message with activation tips
    if IS_WINDOWS:
        act = ".\\.venv\\Scripts\\Activate.ps1"
        shell_hint = f"PowerShell: {act}"
    else:
        act = "source .venv/bin/activate"
        shell_hint = f"bash/zsh: {act}"

    print("\n✅ Environment ready.")
    print(f"To activate: {shell_hint}")
    print("To verify: python -V && pip -V")
    print("To deactivate: 'deactivate'")

if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
