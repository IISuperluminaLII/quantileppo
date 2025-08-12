import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2048,2048, device='cuda'); y = x @ x
    print("Matmul OK, mean:", float(y.mean()))