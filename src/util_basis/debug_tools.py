import torch

def measure_memory(fn, *args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    out = fn(*args, **kwargs)  # run test

    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated()
    
    
    print("Peak memory:", peak_mem / 1024**2, "MB")
    return out