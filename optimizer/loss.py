import torch.nn as nn
import torch.nn.functional as F
import torch
class KLDivLossCustom(nn.Module):
    def __init__(self, temperature=1.0):
        super(KLDivLossCustom, self).__init__()
        self.temperature = temperature

    def forward(self, output_1, output_2):
        # Softmax with temperature scaling
        output_1_soft = F.log_softmax(output_1 / self.temperature, dim=1)
        output_2_soft = F.softmax(output_2 / self.temperature, dim=1)

        # KL Divergence loss
        kl_loss = F.kl_div(output_1_soft, output_2_soft, reduction='batchmean')
        return kl_loss

class NegatedKLDivLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(NegatedKLDivLoss, self).__init__()
        self.kl_div_loss_custom = KLDivLossCustom(temperature=temperature)

    def forward(self, output, output_2):
        kl_div_loss = self.kl_div_loss_custom(output, output_2)
        negated_kl_div_loss = -1 * kl_div_loss
        return negated_kl_div_loss


def reference_loss(model, model_teacher, input, output, reference=4.2):
    # Define the reference value as a torch tensor
    reference_tensor = torch.tensor(reference, dtype=torch.float32, requires_grad=False)
    # Compute the loss as the absolute difference between the reference and the model's output
    loss = torch.abs(nn.CrossEntropyLoss()(model(input), output) - reference_tensor)
    return loss

def base_loss(model, model_teacher, input, output):
    # Compute the loss as the absolute difference between the reference and the model's output
    CE = nn.CrossEntropyLoss()(model(input), output)
    return CE

def KL_retain_loss(model, model_teacher, input, output):
    return KLDivLossCustom()(model(input), model_teacher(input))

def KL_forget_loss(model, model_teacher, input, output):
    return NegatedKLDivLoss()(model(input), model_teacher(input))