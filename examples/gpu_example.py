import torch, time
assert torch.cuda.is_available()
x = torch.randn(8192, 8192, device='cuda')
for _ in range(10):
    y = x @ x
torch.cuda.synchronize()
time.sleep(2)  # keep process alive so nvtop/nvidia-smi can catch it
print(torch.cuda.get_device_name(0))
