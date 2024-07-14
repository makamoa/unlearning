import pytest
import torch
import torch.nn as nn

from optimizer import KLDivLossCustom, NegatedKLDivLoss, InverseKLDivLoss

def create_random_tensors(batch_size, num_classes):
    return torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)

def test_kl_div_loss_custom_is_positive():
    output_1, output_2 = create_random_tensors(10, 5)
    loss_fn = KLDivLossCustom(temperature=2.0)
    loss = loss_fn(output_1, output_2)
    assert isinstance(loss, torch.Tensor), "Output should be a tensor"
    assert loss.item() >= 0, "KL divergence loss should be nonnegative"

def test_negated_kl_div_loss():
    output_1, output_2 = create_random_tensors(10, 5)
    loss_fn = NegatedKLDivLoss(temperature=1.0)
    loss = loss_fn(output_1, output_2)
    assert isinstance(loss, torch.Tensor), "Output should be a tensor"
    assert loss.item() <= 0, \
        "Negated KL divergence loss should be nonpositive"
    
def test_kl_div_loss_custom():
    output_1 = torch.tensor([[1, 1, 1, 1]])
    output_2 = torch.tensor([[10, 10, 10, 10]])
    loss_fn = KLDivLossCustom()
    loss = loss_fn(output_1, output_2)
    assert loss.eq(torch.tensor([[0]])), "Distirbutions must coincide"

def test_inverse_kl_div_loss_custom():
    output_1 = torch.tensor([[1, 1, 1, 1]])
    output_2 = torch.tensor([[10, 10, 10, 10]])
    loss_fn = InverseKLDivLoss(denom_stabilizer=0.01)
    loss = loss_fn(output_1,output_2)
    assert loss.eq(torch.tensor([[100.]])), "Incorrect output"
