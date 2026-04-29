from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F

"""
README FIRST

The below code is a template for the solution. You can change the code according
to your preferences, but the testing function has to save the output of your 
model on the test data as it does in this template. This output must be submitted.

Replace the dummy code with your own code in the TODO sections.

We also encourage you to use tensorboard or wandb to log the training process
and the performance of your model. This will help you to debug your model and
to understand how it is performing. But the template does not include this
functionality.
Link for wandb:
https://docs.wandb.ai/quickstart/
Link for tensorboard: 
https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
"""

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")

# If you have a Mac consult the following link:
# https://pytorch.org/docs/stable/notes/mps.html

# It is important that your model and all data are on the same device.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


def load_data(**kwargs):
    """
    Get the training and test data. The data files are assumed to be in the
    same directory as this script.

    Args:
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - test_data_input: Tensor[N_test_samples, C, H, W]
    where N_train_samples is the number of training samples, N_test_samples is
    the number of test samples, C is the number of channels (1 for grayscale),
    H is the height of the image, and W is the width of the image.
    """
    # Load the training data
    train_data = np.load("train_data.npz")["data"]

    # Make the training data a tensor and normalize to [0, 1]
    train_data = torch.tensor(train_data, dtype=torch.float32) / 255.0

    # Load the test data
    test_data_input = np.load("test_data.npz")["data"]

    # Make the test data a tensor and normalize to [0, 1]
    test_data_input = torch.tensor(test_data_input, dtype=torch.float32) / 255.0

    # The label is the original (complete) image
    train_data_label = train_data.clone()

    # Create masked input: set center 8x8 pixels to 0
    train_data_input = train_data.clone()
    train_data_input[:, :, 10:18, 10:18] = 0.0

    # Visualize the training data if needed
    # Set to False if you don't want to save the images
    if False:
        # Create the output directory if it doesn't exist
        if not Path("train_image_output").exists():
            Path("train_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting train images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.imshow(train_data_input[i].squeeze(), cmap="gray")
            plt.title("Training Input")
            plt.subplot(1, 2, 2)
            plt.title("Training Label")
            plt.imshow(train_data_label[i].squeeze(), cmap="gray")

            plt.savefig(f"train_image_output/image_{i}.png")
            plt.close()

    return train_data_input, train_data_label, test_data_input


def training(train_data_input, train_data_label, **kwargs):
    """
    Train the model. Fill in the details of the data loader, the loss function,
    the optimizer, and the training loop.

    Args:
    - train_data_input: Tensor[N_train_samples, C, H, W]
    - train_data_label: Tensor[N_train_samples, C, H, W]
    - kwargs: Additional arguments that you might find useful - not necessary

    Returns:
    - model: torch.nn.Module
    """
    model = Model()
    model.train()
    model.to(device)

    # MSE loss - matches the evaluation metric
    criterion = nn.MSELoss()

    # Adam optimizer with a good learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    batch_size = 128
    dataset = TensorDataset(train_data_input, train_data_label)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True)

    # Training loop
    n_epochs = 50

    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in tqdm(
            data_loader, desc=f"Training Epoch {epoch}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)

            # Compute loss on the full image for better context,
            # but weight the masked region more heavily
            loss_full = criterion(output, y)
            # Extra loss on the masked center region where evaluation happens
            loss_center = criterion(
                output[:, :, 10:18, 10:18], y[:, :, 10:18, 10:18]
            )
            # Combined loss: emphasize the center region
            loss = 0.3 * loss_full + 0.7 * loss_center

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)

        print(f"Epoch {epoch} - avg loss: {avg_loss:.6f}, "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_model_state)
    return model


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    """
    U-Net style encoder-decoder for image inpainting.
    The encoder extracts features from the masked image,
    and the decoder reconstructs the full image with skip connections.
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2, 2)       # 28 -> 14

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, 2)       # 14 -> 7

        self.enc3 = ConvBlock(128, 256)

        # Decoder
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 7 -> 14
        self.dec2 = ConvBlock(256, 128)  # 128 (up) + 128 (skip) = 256

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 14 -> 28
        self.dec1 = ConvBlock(128, 64)   # 64 (up) + 64 (skip) = 128

        # Output layer - sigmoid to keep output in [0, 1]
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)       # [B, 64, 28, 28]
        p1 = self.pool1(e1)     # [B, 64, 14, 14]

        e2 = self.enc2(p1)      # [B, 128, 14, 14]
        p2 = self.pool2(e2)     # [B, 128, 7, 7]

        e3 = self.enc3(p2)      # [B, 256, 7, 7]

        # Decoder path with skip connections
        d2 = self.up2(e3)       # [B, 128, 14, 14]
        d2 = torch.cat([d2, e2], dim=1)  # [B, 256, 14, 14]
        d2 = self.dec2(d2)      # [B, 128, 14, 14]

        d1 = self.up1(d2)       # [B, 64, 28, 28]
        d1 = torch.cat([d1, e1], dim=1)  # [B, 128, 28, 28]
        d1 = self.dec1(d1)      # [B, 64, 28, 28]

        out = self.out_conv(d1) # [B, 1, 28, 28]
        out = torch.sigmoid(out)
        return out


def testing(model, test_data_input):
    """
    Uses your model to predict the ouputs for the test data. Saves the outputs
    as a binary file. This file needs to be submitted. This function does not
    need to be modified except for setting the batch_size value. If you choose
    to modify it otherwise, please ensure that the generating and saving of the
    output data is not modified.

    Args:
    - model: torch.nn.Module
    - test_data_input: Tensor
    """
    model.eval()
    model.to(device)

    with torch.no_grad():
        test_data_input = test_data_input.to(device)
        # Predict the output batch-wise to avoid memory issues
        test_data_output = []
        batch_size = 256
        for i in tqdm(
            range(0, test_data_input.shape[0], batch_size),
            desc="Predicting test output",
        ):
            output = model(test_data_input[i : i + batch_size])
            test_data_output.append(output.cpu())
        test_data_output = torch.cat(test_data_output)

    # Scale back to [0, 255] since we normalized to [0, 1]
    test_data_output = test_data_output * 255.0

    # Ensure the output has the correct shape
    assert test_data_output.shape == test_data_input.shape, (
        f"Expected shape {test_data_input.shape}, but got "
        f"{test_data_output.shape}."
        "Please ensure the output has the correct shape."
        "Without the correct shape, the submission cannot be evaluated and "
        "will hence not be valid."
    )

    # Save the output
    test_data_output = test_data_output.numpy()
    # Ensure all values are in the range [0, 255]
    save_data_clipped = np.clip(test_data_output, 0, 255)
    # Convert to uint8
    # Ensure your model outputs values in the [0, 255] range before this step! If you normalized your data to [0, 1], you must multiply by 255 before saving.
    save_data_uint8 = save_data_clipped.astype(np.uint8)
    # Loss is only computed on the masked area - so set the rest to 0 to save
    # space
    save_data = np.zeros_like(save_data_uint8)
    save_data[:, :, 10:18, 10:18] = save_data_uint8[:, :, 10:18, 10:18]

    np.savez_compressed(
        "submit_this_test_data_output.npz", data=save_data)

    # You can plot the output if you want
    # Set to False if you don't want to save the images
    if True:
        # Create the output directory if it doesn't exist
        if not Path("test_image_output").exists():
            Path("test_image_output").mkdir()
        for i in tqdm(range(20), desc="Plotting test images"):
            # Show the training and the target image side by side
            plt.subplot(1, 2, 1)
            plt.title("Test Input")
            plt.imshow(test_data_input[i].squeeze().cpu().numpy(), cmap="gray")
            plt.subplot(1, 2, 2)
            plt.imshow(test_data_output[i].squeeze(), cmap="gray")
            plt.title("Test Output")

            plt.savefig(f"test_image_output/image_{i}.png")
            plt.close()


def main():
    seed = 0
    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # You don't need to change the code below
    # Load the data
    train_data_input, train_data_label, test_data_input = load_data()
    # Train the model
    model = training(train_data_input, train_data_label)

    # Test the model (this also generates the submission file)
    # The name of the submission file is submit_this_test_data_output.npz
    testing(model, test_data_input)

    return None


if __name__ == "__main__":
    main()
