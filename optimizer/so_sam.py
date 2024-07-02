import torch

class SO_SAM(torch.optim.Optimizer):
    """
    Initializes a handler for a second-order Sharpness-Aware Minimization
    (SAM) optimizer.
    
    Args:
        params: Parameters to be optimizer.
        base_optimizer (torch.optim.Optimizer): The optimizer on which
            SAM is applied.
        rho (float): The `rho` parameter of SAM.
        precision (float): The precision used for estimating
            the Hessian-gradient product.
        kwargs: Additional hyperparameters for the `base_optimizer`.
    """
    def __init__(self, params, base_optimizer, rho=0.05, 
                 precision=1e-10, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, precision=precision, **kwargs)
        super(SO_SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=True):
        grad_norm = self._grad_norm()
        self.state['first_grad_norm'] = grad_norm
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.clone().detach()
                self.state[p]["first_grad"] = p.grad.clone().detach()
                e_w = p.grad * scale.to(p).type_as(p)
                p.add_(e_w)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=True):
        for group in self.param_groups:
            scale = group["precision"]

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["second_grad"] = p.grad.clone().detach()
                p.copy_(self.state[p]["old_p"] + p.grad * scale)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def third_step(self, zero_grad=True):
        device = self.param_groups[0]["params"][0].device
        scalar_product = torch.zeros(1, device=device)
        for group in self.param_groups:
            scale = group["rho"] / \
                (self.state['first_grad_norm'] * group["precision"] + 1e-18)
            for p in group["params"]:
                if p.grad is None: continue
                hessian_grad = p.grad - self.state[p]["first_grad"]
                scalar_product.add_(
                    torch.dot(
                        hessian_grad.view(-1, 1).squeeze(),
                        self.state[p]["first_grad"].view(-1, 1).squeeze()))
                p.grad.copy_(self.state[p]["second_grad"] + \
                    scale.to(p) * hessian_grad)
        for group in self.param_groups:
            scale = group["rho"] * scalar_product / \
                (self.state['first_grad_norm'].pow(3) * group["precision"] + 1e-18)
            for p in group["params"]:
                if p.grad is None: continue
                p.grad.sub_(self.state[p]["first_grad"] * scale.to(p))
                p.copy_(self.state[p]["old_p"])

        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.linalg.norm(
                    torch.stack([
                        (p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups 
                        for p in group["params"]
                        if p.grad is not None
                    ]),
                    ord=2
               )
        return norm