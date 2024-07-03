import torch
import pytest

from optimizer import SO_SAM

def test_so_sam_initialization():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Linear(10, 2).to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SO_SAM(model.parameters(), base_optimizer, rho=0.1, lr=0.01)
    assert optimizer is not None
    assert optimizer.defaults['rho'] == 0.1
    assert optimizer.defaults['lr'] == 0.01

def test_so_sam_first_step():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Linear(10, 2).to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SO_SAM(model.parameters(), base_optimizer, rho=0.1, lr=0.01)
    
    input = torch.randn(10).to(device)
    target = torch.tensor([1.0, 0.0]).to(device)
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    
    loss.backward()
    
    optimizer.first_step()
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                assert p.data.ne(optimizer.state[p]['old_p']).any()

def test_so_sam_second_step():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Linear(10, 2).to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SO_SAM(model.parameters(), base_optimizer, rho=0.1, lr=0.01)
    
    input = torch.randn(10).to(device)
    target = torch.tensor([1.0, 0.0]).to(device)
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    optimizer.first_step()
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    optimizer.second_step()
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                assert p.data.ne(optimizer.state[p]['old_p']).any()

def test_sam_third_step():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.Linear(10, 2).to(device)
    base_optimizer = torch.optim.SGD
    optimizer = SO_SAM(model.parameters(), base_optimizer, rho=0.1, lr=0.01)
    
    input = torch.randn(10).to(device)
    target = torch.tensor([1.0, 0.0]).to(device)
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    optimizer.first_step()
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    optimizer.second_step()
    
    output = model(input)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    
    optimizer.third_step()
    
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                assert p.data.ne(optimizer.state[p]['old_p']).any()