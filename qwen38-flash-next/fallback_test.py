import torch
free, total = torch.cuda.mem_get_info()
target = free + 6 * 2**30
chunks, got = [], 0
try:
    while got < target:
        chunks.append(torch.empty(2 * 2**30, dtype=torch.uint8, device="cuda"))
        got += 2 * 2**30
    print(f"FALLBACK ON: allocated {got / 2**30:.0f} GiB with only {free / 2**30:.1f} GiB free VRAM")
except torch.OutOfMemoryError:
    print(f"FALLBACK OFF: OOM after {got / 2**30:.0f} GiB ({free / 2**30:.1f} GiB was free)")
