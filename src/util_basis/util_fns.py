import torch

# calculate loss between two lists (like (array1-array2)^2))
def mse_loss_over_list(list1, list2,mask=None):
    if mask == None:
        loss = sum(torch.mean((a - b)**2) for a, b in zip(list1, list2))
        return loss / len(list1)
    else:
        loss = 0.
        for a, b, m in zip(list1, list2, mask):
            diff = (a - b)**2
            loss += torch.mean(diff[m])
        return loss / len(list1)

def torch_lexsort(keys, descending=False):
    """
    Lexicographically sort by multiple keys (last key has highest priority).
    Equivalent to np.lexsort(keys).
    keys: list/tuple of 1D tensors.
    """
    assert len(keys) > 0
    idx = torch.arange(keys[0].shape[0], device=keys[0].device)
    for key in keys[:-1]:
        _, idx = torch.sort(key[idx], descending=descending)
        idx = idx[idx]
    _, idx = torch.sort(keys[-1][idx], descending=descending)
    return idx