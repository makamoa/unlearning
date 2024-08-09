import torch

def calculate_accuracy(output: torch.Tensor,
                       target: torch.Tensor,
                       topk: tuple=(1,)) -> list:
    """
    Calculate the top-k accuracy between the given output and target labels
    for all values of `k` specified in `topk`.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res