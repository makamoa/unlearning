import torch.nn as nn
import torch.nn.functional as F

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
    
class InverseKLDivLoss(nn.Module):
    def __init__(self, regularizer=1.0, denom_stabilizer=1e-1, temperature=1.0):
        assert isinstance(regularizer, float), "The wrong value type"
        assert isinstance(denom_stabilizer, float), "The wrong value type"
        assert isinstance(temperature, float), "The wrong value type"
        super(InverseKLDivLoss, self).__init__()
        self.KLDivLoss = KLDivLossCustom(temperature)
        self.regularizer = regularizer
        self.denom_stabilizer = denom_stabilizer

    def forward(self, output_1, output_2):
        kl_loss = self.KLDivLoss(output_1, output_2)
        return self.regularizer / (kl_loss + self.denom_stabilizer)

def base_loss(model, model_teacher, input, output):
    return nn.CrossEntropyLoss()(model(input), output)

def KL_retain_loss(model, model_teacher, input, output):
    return KLDivLossCustom()(model(input), model_teacher(input))

def KL_forget_loss(model, model_teacher, input, output):
    return NegatedKLDivLoss()(model(input), model_teacher(input))

def inverse_KL_forget_loss(model, model_teacher, input, output):
    return InverseKLDivLoss(regularizer=0.1, denom_stabilizer=0.2)(model(input), model_teacher(input))