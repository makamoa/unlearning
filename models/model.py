import torch
import torch.nn as nn
import torchvision.models as models
import timm

def get_model(model_name, num_classes=100, pretrained_weights=None, weight_path=None):
    assert pretrained_weights is None or weight_path is None, \
        "Specify only one: either 'pretrained_weights' or 'weight_path', not both."
    if model_name == 'resnet18':
        model = models.resnet18(weights=pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=pretrained_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pweights=pretrained_weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'resnest50d':
        model = timm.create_model('resnest50d', weights=pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnest101e':
        model = timm.create_model('resnest101e', weights=pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'binary_classifier':
        model = SimpleBinaryClassifier(input_size=num_classes)
    elif model_name == 'contrastive_classifier':
        model = ComparisonModel(input_dim=num_classes, use_linear=True)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print(f"Loaded weights from {weight_path}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))

    print(f"Model '{model_name}' loaded successfully!")
    print(f"Number of layers: {num_layers}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    return model


def negative_loss(loss_fn):
    """
    Returns a new loss function that computes the negative of the given loss function.

    Args:
        loss_fn (function): A PyTorch loss function.

    Returns:
        function: A new loss function that computes the negative of the input loss function.
    """

    def neg_loss(output, target):
        return -loss_fn(output, target)

    return neg_loss

class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleBinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)

        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        x = torch.sigmoid(self.fc3(x))
        return x

# Define the Comparison Model (Classifier)
class ComparisonModel(nn.Module):
    def __init__(self, input_dim, feature_size=None, temperature=1.0, use_linear=True, use_norm=True):
        super(ComparisonModel, self).__init__()
        self.temperature = temperature
        self.use_linear = use_linear
        self.use_norm = use_norm
        if feature_size is None:
            feature_size = input_dim // 2

        if self.use_linear:
            self.linear_transform = nn.Linear(input_dim, feature_size, dtype=torch.float32)
            if self.use_norm:
                self.normalize = nn.LayerNorm(feature_size, dtype=torch.float32)
        elif self.use_norm:
            self.normalize = nn.LayerNorm(input_dim, dtype=torch.float32)

        self.sigmoid = nn.Sigmoid()

    def forward(self, v1, v2):
        if self.use_linear:
            v1 = self.linear_transform(v1 - v2)
            #v2 = self.linear_transform(v1 - v2)

        if self.use_norm:
            v1 = self.normalize(v1)
            #v2 = self.normalize(v2)

        # Compute the inner product (dot product) between the two vectors
        inner_product = torch.mean(v1, dim=1, keepdim=True) / self.temperature

        # Apply the sigmoid function
        output = self.sigmoid(inner_product)
        return output


if __name__ == '__main__':
    model_names = ['resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b7', 'resnest50d', 'resnest101e']
    for model_name in model_names:
        print(f"\nLoading model: {model_name}")
        model = get_model(model_name, num_classes=100)