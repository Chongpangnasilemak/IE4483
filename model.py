import torch
import torch.nn as nn
import torchvision

# Create an VGG11 feature extractor
def create_vgg11(device,class_names):
    """
    Create a VGG11 feature extractor model.

    This function constructs a VGG11 model pre-trained on ImageNet, 
    which can be used to extract features from input images. The final 
    classification layer is modified to accommodate the specified 
    number of class names.

    Args:
        device (str): The device on which the model will be loaded (e.g., 'cpu','mps', 'cuda').
        class_names (int): The number of output classes for the classifier layer, 
                           corresponding to the specific task at hand.

    Returns:
        torch.nn.Module: A modified VGG11 model ready for feature extraction, 
                         with the final layer adjusted to the specified number of classes.
    """
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.VGG11_Weights.DEFAULT
    model = torchvision.models.vgg11(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)        
        
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=len(class_names)),
    ).to(device)

    # 5. Give the model a name
    model.name = "VGG11"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an VGG16 feature extractor
def create_vgg16(device,class_names):
    """
    Create a VGG16 feature extractor model.

    This function constructs a VGG16 model pre-trained on ImageNet, 
    which can be used to extract features from input images. The final 
    classification layer is modified to accommodate the specified 
    number of class names.

    Args:
        device (str): The device on which the model will be loaded (e.g., 'cpu','mps', 'cuda').
        class_names (int): The number of output classes for the classifier layer, 
                           corresponding to the specific task at hand.

    Returns:
        torch.nn.Module: A modified VGG16 model ready for feature extraction, 
                         with the final layer adjusted to the specified number of classes.
    """
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.VGG16_Weights.DEFAULT
    model = torchvision.models.vgg16(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)        
        
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=len(class_names)),
    ).to(device)
 
    # 5. Give the model a name
    model.name = "VGG16"
    print(f"[INFO] Created new {model.name} model.")
    return model

# Create an VGG19 feature extractor
def create_vgg19(device,class_names):
    """
    Create a VGG11 feature extractor model.

    This function constructs a VGG19 model pre-trained on ImageNet, 
    which can be used to extract features from input images. The final 
    classification layer is modified to accommodate the specified 
    number of class names.

    Args:
        device (str): The device on which the model will be loaded (e.g., 'cpu','mps', 'cuda').
        class_names (int): The number of output classes for the classifier layer, 
                           corresponding to the specific task at hand.

    Returns:
        torch.nn.Module: A modified VGG19 model ready for feature extraction, 
                         with the final layer adjusted to the specified number of classes.
    """
    # 1. Get the base model with pretrained weights and send to target device
    weights = torchvision.models.VGG19_Weights.DEFAULT
    model = torchvision.models.vgg19(weights=weights).to(device)

    # 2. Freeze the base model layers
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Set the seeds
    torch.manual_seed(42)        
        
    # 4. Change the classifier head
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=4096),
        nn.Dropout(p=0.5,inplace=False),
        nn.Linear(in_features=4096, out_features=len(class_names)),
    ).to(device)
 
    # 5. Give the model a name
    model.name = "VGG19"
    print(f"[INFO] Created new {model.name} model.")
    return model