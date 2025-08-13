import os
import sys

# Path where you'll put the required DLLs
dll_dir = os.path.abspath("C:/Users/Shashank Murthy/.conda/envs/quantileppo/Lib/site-packages/atari_py")

# Create the folder if it doesn't exist
os.makedirs(dll_dir, exist_ok=True)

# Add to PATH so ctypes can find them
os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")

print(f"[INFO] Added {dll_dir} to PATH")

# Optional: verify
if dll_dir not in os.environ["PATH"].split(os.pathsep):
    print("[WARN] DLL dir not on PATH!")

# Now you can safely import atari_py (if the DLLs are present)
try:
    import atari_py
    print("[INFO] atari_py imported successfully")
except Exception as e:
    print(f"[ERROR] atari_py still failed: {e}")