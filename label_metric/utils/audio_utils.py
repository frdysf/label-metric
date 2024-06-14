import torch
import torch.nn.functional as F

def standardize_duration(y: torch.Tensor, 
                         sr: int, 
                         dur: float = 1.0) -> torch.Tensor:
    n_samples = y.shape[1]
    target_samples = int(sr * dur)
    if n_samples > target_samples:
        y = y[:,:target_samples]
    else:
        y = F.pad(y, (0, target_samples - n_samples))
    assert y.shape[1] == target_samples
    return y

if __name__ == '__main__':
    x = torch.ones(1,4)
    sr = 5
    print(standardize_duration(x, sr))