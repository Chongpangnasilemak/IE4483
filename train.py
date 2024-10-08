import os
import torch
import engine, model, utils
from utils import *
from data_setup import create_dataloaders
import torchvision

def main():
    # Setup directories
    train_dir = "datasets/train"
    val_dir = "datasets/val"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Create auto-transforms
    transforms = torchvision.models.VGG19_Weights.DEFAULT.transforms() 


    # Create DataLoaders
    train_dataloader, test_dataloader, val_dataloader, class_names = create_dataloaders(
        train_dir=train_dir,
        val_dir=val_dir,
        transform=transforms,
        batch_size=int(os.getenv('BATCH_SIZE')),
    )

    # Create vgg model
    model_vgg19 = model.create_vgg19(device=device,class_names=class_names).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_vgg19.parameters(),
                                lr=float(os.getenv('LEARNING_RATE')))

    # Training 
    engine.train(model=model_vgg19,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                val_dataloader=val_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=int(os.getenv('NUM_EPOCHS')),
                device=device,
                )

    # Save the model
    utils.save_model(model=model_vgg19,
                    target_dir="models",
                    model_name="vgg19_0.pth")

if __name__ == '__main__':
    main()
