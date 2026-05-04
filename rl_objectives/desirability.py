import torch


def _to_tensor(x):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32)


def higher_is_better(x, low, high):
    x = _to_tensor(x)
    return ((x - low) / max(high - low, 1e-8)).clamp(0.0, 1.0)


def lower_is_better(x, good, bad):
    x = _to_tensor(x)
    return ((bad - x) / max(bad - good, 1e-8)).clamp(0.0, 1.0)


def interval_desirability(x, low_hard, low_soft, high_soft, high_hard):
    x = _to_tensor(x)
    out = torch.zeros_like(x, dtype=torch.float32)
    out = torch.where((x >= low_soft) & (x <= high_soft), torch.ones_like(out), out)
    left = (x > low_hard) & (x < low_soft)
    out = torch.where(left, ((x - low_hard) / max(low_soft - low_hard, 1e-8)).clamp(0, 1), out)
    right = (x > high_soft) & (x < high_hard)
    out = torch.where(right, ((high_hard - x) / max(high_hard - high_soft, 1e-8)).clamp(0, 1), out)
    return out


def gaussian(x, mu, sigma):
    x = _to_tensor(x)
    sigma = max(float(sigma), 1e-8)
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def max_gaussian(x, mu, sigma):
    x = _to_tensor(x)
    return torch.where(x <= mu, torch.ones_like(x), gaussian(x, mu, sigma))


def min_gaussian(x, mu, sigma):
    x = _to_tensor(x)
    return torch.where(x >= mu, torch.ones_like(x), gaussian(x, mu, sigma))


def thresholded(x, threshold):
    x = _to_tensor(x)
    t = max(float(threshold), 1e-8)
    return torch.where(x >= t, torch.ones_like(x), (x / t).clamp(0.0, 1.0))


def _stack_components(component_scores):
    keys = list(component_scores.keys())
    vals = torch.stack([_to_tensor(component_scores[k]).float() for k in keys], dim=1)
    return keys, vals


def _weights(keys, weights, device):
    if weights is None:
        w = torch.ones(len(keys), device=device)
    else:
        w = torch.tensor([float(weights.get(k, 1.0)) for k in keys], device=device)
    return w / w.sum().clamp_min(1e-8)


def weighted_geometric_mean(component_scores, weights=None, eps=1e-8):
    keys, vals = _stack_components(component_scores)
    w = _weights(keys, weights, vals.device)
    return torch.exp((torch.log(vals.clamp_min(eps)) * w.view(1, -1)).sum(dim=1))


def weighted_linear_sum(component_scores, weights=None):
    keys, vals = _stack_components(component_scores)
    w = _weights(keys, weights, vals.device)
    return (vals * w.view(1, -1)).sum(dim=1)


def tchebycheff_score(component_scores, weights=None):
    keys, vals = _stack_components(component_scores)
    w = _weights(keys, weights, vals.device)
    return 1.0 - (w.view(1, -1) * (1.0 - vals)).max(dim=1).values
