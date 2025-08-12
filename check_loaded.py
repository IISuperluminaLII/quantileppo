
import psutil, torch, time
p = psutil.Process()
# trigger torch import to load DLLs
_ = torch.__version__
hits = [m.path for m in p.memory_maps() if 'libiomp5md.dll' in m.path.lower() or 'libomp' in m.path.lower()]
print("OpenMP DLLs:")
for h in sorted(set(hits)): print(" -", h)
time.sleep(0.5)
