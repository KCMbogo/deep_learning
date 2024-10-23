import os
import pandas as pd
from data import *
from tqdm.auto import tqdm
from model_0 import TinyVGG, train, model_0_results
from helper_fn import train_step, test_step, plot_loss_curves
from timeit import default_timer as timer 
from typing import List

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor() 
])

# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

# Turn Datasets into DataLoader's
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()

torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented, 
                                        batch_size=BATCH_SIZE, 
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple, 
                                    batch_size=BATCH_SIZE, 
                                    shuffle=False, 
                                    num_workers=NUM_WORKERS)

# Create model_1 and send it to the target device
torch.manual_seed(42)
model_1 = TinyVGG(
    in_shape=3,
    hidden_units=10,
    out_shape=len(train_data_augmented.classes))

# print(model_1)


# Set random seeds
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 5

# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)


start_time = timer()

# Train model_1
model_1_results = train(model=model_1, 
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn, 
                        epochs=NUM_EPOCHS)

end_time = timer()
print(f"Total training time: {end_time-start_time:.3f} seconds")

# plot_loss_curves(model_1_results)

model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)

# Setup a plot 
# plt.figure(figsize=(15, 10))

# Get number of epochs
# epochs = range(len(model_0_df))

# Plot train loss
# plt.subplot(2, 2, 1)
# plt.plot(epochs, model_0_df["train_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["train_loss"], label="Model 1")
# plt.title("Train Loss")
# plt.xlabel("Epochs")
# plt.legend()

# Plot test loss
# plt.subplot(2, 2, 2)
# plt.plot(epochs, model_0_df["test_loss"], label="Model 0")
# plt.plot(epochs, model_1_df["test_loss"], label="Model 1")
# plt.title("Test Loss")
# plt.xlabel("Epochs")
# plt.legend()

# Plot train accuracy
# plt.subplot(2, 2, 3)
# plt.plot(epochs, model_0_df["train_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["train_acc"], label="Model 1")
# plt.title("Train Accuracy")
# plt.xlabel("Epochs")
# plt.legend()

# Plot test accuracy
# plt.subplot(2, 2, 4)
# plt.plot(epochs, model_0_df["test_acc"], label="Model 0")
# plt.plot(epochs, model_1_df["test_acc"], label="Model 1")
# plt.title("Test Accuracy")
# plt.xlabel("Epochs")
# plt.legend();

# plt.show()


# Download custom image
import requests

# Setup custom image path
custom_image_path = data_path / "04-pizza-dad.jpeg"

# Download the image if it doesn't already exist
if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        # When downloading from GitHub, need to use the "raw" file link
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")
    
    
custom_image_uint8 = torchvision.io.read_image(str(custom_image_path)) # dtype: uint8
custom_image = custom_image_uint8.type(torch.float32)
custom_image = custom_image / 255 # turn the values to range 0 - 1

# Print out image data
# print(f"Custom image tensor:\n{custom_image}\n")
# print(f"Custom image shape: {custom_image.shape}\n")
# print(f"Custom image dtype: {custom_image.dtype}")

# # Plot custom image
# plt.imshow(custom_image.permute(1, 2, 0)) # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
# plt.title(f"Image shape: {custom_image.shape}")
# plt.axis(False);
# plt.show()


# Create transform pipleine to resize image
custom_image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
])


def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str, 
                        class_names: List[str] = None, 
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""
    
    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    
    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 
    
    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)
    
    # 4. Make sure the model is on the target device
    model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)
    
        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))
        
    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()
    
    
# Pred on our custom image
pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)