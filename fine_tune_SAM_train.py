import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
from patchify import patchify  #Only to handle large images
import random
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import SamModel, SamProcessor
from tqdm import tqdm
from statistics import mean
from datasets import Dataset
import torch
from torch.optim import Adam
import monai
from torch.utils.data import DataLoader
import json
import skimage.draw
import skimage.io
from tqdm import tqdm

source_img_path = "training/wsis"
training_img_path = 'training/wsi_pngs'  # Path to your downscaled PNG files
annotations_path = "training/annotations/100cohort"
training_mask_path = "training/masks"
model_checkpoint = "model_checkpoint.pth"

# This is the multiplier by which the source images are downscaled to training images
# Decrease this if the training image resolutions are too low
# NOTE: This same downscale factor will be applied to the json annotations as well
downscale_factor = 16 

# Pre-processing - rename bad file names
def rename_files(directory):
    for filename in os.listdir(directory):
        # Check if there's a space in the filename
        if ' ' in filename:
            # Create the new filename by replacing spaces with underscores
            new_filename = filename.replace(' ', '_')
            # Get the full path of the old and new filenames
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {filename} to {new_filename}')

rename_files(source_img_path)
rename_files(annotations_path)
rename_files(training_mask_path)


# Pre-processing - downscale source images (NDPI) to easily processable images (PNG)

# Function to convert NDPI images to downscaled PNG
def convert_ndpi_to_png(ndpi_path, png_output_path, downscale_factor):
    # Ensure the output directory exists
    if not os.path.exists(png_output_path):
        os.makedirs(png_output_path)

    # Get all the NDPI files
    ndpi_files = [f for f in os.listdir(ndpi_path) if f.endswith('.ndpi')]

    for ndpi_file in tqdm(ndpi_files, desc="Converting NDPI to PNG"):
        ndpi_file_path = os.path.join(ndpi_path, ndpi_file)

        # Open the NDPI file
        slide = openslide.OpenSlide(ndpi_file_path)
        # Calculate the downscaled dimensions
        width, height = slide.level_dimensions[0]
        new_width = width // downscale_factor
        new_height = height // downscale_factor

        # Get the thumbnail (downscaled image)
        thumbnail = slide.get_thumbnail((new_width, new_height))

        # Save the thumbnail as PNG
        base_name = os.path.splitext(ndpi_file)[0]
        png_file_path = os.path.join(png_output_path, f"{base_name}.ndpi.png")
        thumbnail.save(png_file_path, "PNG")

        print(f"Saved {png_file_path}")

convert_ndpi_to_png(source_img_path, training_img_path, downscale_factor)

# Pre-processing - convert JSON annotations to mask images
def create_masks_from_json(png_path, json_path, mask_output_path, downscale_factor):
    # Ensure the output directory exists
    if not os.path.exists(mask_output_path):
        os.makedirs(mask_output_path)

    # Get all JSON files
    json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]

    for json_file in tqdm(json_files, desc="Creating masks from JSON"):
        json_file_path = os.path.join(json_path, json_file)
        
        # Load JSON annotations
        with open(json_file_path, 'r') as f:
            annotations = json.load(f)

        # Find the corresponding PNG file
        base_name = json_file.replace('.json', '')
        png_file_path = os.path.join(png_path, f"{base_name}.png")

        if os.path.exists(png_file_path):
            # Open the PNG file to get dimensions
            image = skimage.io.imread(png_file_path)
            height, width = image.shape[:2]

            # Create an empty mask as a numpy array
            mask_np = np.zeros((height, width), dtype=np.uint8)

            for ann in annotations:  # Assuming annotations is a list of features
                if ann['geometry']['type'] == "Polygon":
                    # Extract segmentation polygon
                    for seg in ann['geometry']['coordinates']:
                        # Convert polygons to a binary mask and add it to the main mask
                        poly = np.array(seg) // downscale_factor  # Downscale the coordinates
                        if np.any(poly >= np.array([width, height])) or np.any(poly < 0):
                            print(f"Polygon coordinates out of bounds in {json_file}")
                            continue
                        rr, cc = skimage.draw.polygon(poly[:, 1], poly[:, 0], mask_np.shape)
                        
                        mask_np[rr, cc] = 255

                        # Debug: Print some of the polygon coordinates
                    print(f"Drawing polygon with coordinates: {poly}")

            # Debug: Print non-zero values in the mask
            non_zero_values = np.count_nonzero(mask_np)
            print(f"Non-zero values in the mask for {json_file}: {non_zero_values}")

            # Save the numpy array as a PNG file
            mask_file_path = os.path.join(mask_output_path, f"{base_name}.png")
            skimage.io.imsave(mask_file_path, mask_np)

            print(f"Saved mask for {json_file} to {mask_file_path}")
        else:
            print(f"PNG file {png_file_path} not found. Skipping.")

# Create masks from JSON annotations
create_masks_from_json(training_img_path, annotations_path, training_mask_path, downscale_factor)


# Now. let us divide these large images into smaller patches for training. We can use patchify or write custom code.
def match_image_and_mask_filenames(image_folder, mask_folder):
    image_filenames = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    mask_filenames = [f for f in os.listdir(mask_folder) if f.endswith(".png")]
    matched_pairs = [(img, img) for img in image_filenames if img in mask_filenames]
    return matched_pairs

def load_and_patchify_images_and_masks(image_folder, mask_folder, patch_size, step):
    images = []
    masks = []
    matched_pairs = match_image_and_mask_filenames(image_folder, mask_folder)

    for image_filename, mask_filename in matched_pairs:
        # Load and patchify image
        img_path = os.path.join(image_folder, image_filename)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        patches_img = patchify(img, (patch_size, patch_size, 3), step=step)  # Ensure RGB output
        patches_img = patches_img.reshape(-1, patch_size, patch_size, 3)  # Reshape to remove singleton dimensions
        images.extend(patches_img)

        # Load and patchify mask
        mask_path = os.path.join(mask_folder, mask_filename)
        mask = Image.open(mask_path).convert("L")
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=-1)  # Ensure mask has a channel dimension

        patches_mask = patchify(mask, (patch_size, patch_size, 1), step=step)  # Assume single-channel mask
        patches_mask = patches_mask.reshape(-1, patch_size, patch_size, 1)  # Reshape to remove singleton dimensions
        masks.extend(patches_mask)

    return np.array(images), np.array(masks)

patch_size = 256
step = 256

# Load and patchify images and masks
images, masks = load_and_patchify_images_and_masks(training_img_path, training_mask_path, patch_size, step)
print(f"Image patches shape: {images.shape}")
print(f"Mask patches shape: {masks.shape}")


# Now, let us delete empty masks as they may cause issues later on during training. If a batch contains empty masks then the loss function will throw an error as it may not know how to handle empty tensors.
# Create a list to store the indices of non-empty masks
valid_indices = [i for i, mask in enumerate(masks) if mask.max() != 0]
# Filter the image and mask arrays to keep only the non-empty pairs
filtered_images = images[valid_indices]
filtered_masks = masks[valid_indices]
print("Image shape:", filtered_images.shape)  # e.g., (num_frames, height, width, num_channels)
print("Mask shape:", filtered_masks.shape)


# Let us create a 'dataset' that serves us input images and masks for the rest of our journey.
# Convert the NumPy arrays to Pillow images and preprocess them
dataset_dict = {
    "image": [Image.fromarray(img) for img in filtered_images],
    "label": [Image.fromarray(mask.squeeze()) for mask in filtered_masks],  # Use squeeze to remove the singleton dimension
}

# Create the dataset using the datasets.Dataset class
dataset = Dataset.from_dict(dataset_dict)

# Function to print the number of channels in the first image
def print_number_of_channels(dataset):
    if len(dataset) > 0:
        item = dataset[0]
        image = item['image']
        # Convert to NumPy array if necessary
        if isinstance(image, Image.Image):
            image = np.array(image)
        num_channels = image.shape[-1] if image.ndim == 3 else 1
        print(f"The first image has {num_channels} channel(s)")
    else:
        print("The dataset is empty")

# Print the number of channels for the first image in the dataset
print_number_of_channels(dataset)

img_num = random.randint(0, filtered_images.shape[0]-1)
example_image = dataset[img_num]["image"]
example_mask = dataset[img_num]["label"]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Plot the first image on the left
axes[0].imshow(np.array(example_image), cmap='gray')  # Assuming the first image is grayscale
axes[0].set_title("Image")

# Plot the second image on the right
axes[1].imshow(example_mask, cmap='gray')  # Assuming the second image is grayscale
axes[1].set_title("Mask")

# Hide axis ticks and labels
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

# Display the images side by side
plt.show()

#Get bounding boxes from mask.
def get_bounding_box(ground_truth_map):
  # get bounding box from mask
  y_indices, x_indices = np.where(ground_truth_map > 0)
  x_min, x_max = np.min(x_indices), np.max(x_indices)
  y_min, y_max = np.min(y_indices), np.max(y_indices)
  # add perturbation to bounding box coordinates
  H, W = ground_truth_map.shape
  x_min = max(0, x_min - np.random.randint(0, 20))
  x_max = min(W, x_max + np.random.randint(0, 20))
  y_min = max(0, y_min - np.random.randint(0, 20))
  y_max = min(H, y_max + np.random.randint(0, 20))
  bbox = [x_min, y_min, x_max, y_max]

  return bbox

from torch.utils.data import Dataset

class SAMDataset(Dataset):
  """
  This class is used to create a dataset that serves input images and masks.
  It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
  """
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    item = self.dataset[idx]
    image = item["image"]
    ground_truth_mask = np.array(item["label"])

    # get bounding box prompt
    prompt = get_bounding_box(ground_truth_mask)

    # prepare image and prompt for the model
    inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

    # remove batch dimension which the processor adds by default
    inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
    inputs["ground_truth_mask"] = ground_truth_mask

    return inputs

# Initialize the processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the SAMDataset
train_dataset = SAMDataset(dataset=dataset, processor=processor)

example = train_dataset[0]
for k,v in example.items():
  print(k,v.shape)

# Create a DataLoader instance for the training dataset
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

batch = next(iter(train_dataloader))
for k,v in batch.items():
  print(k,v.shape)

batch["ground_truth_mask"].shape

# Load the model
model = SamModel.from_pretrained("facebook/sam-vit-base")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
  if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
    param.requires_grad_(False)

# Initialize the optimizer and the loss function
optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

#Training loop
num_epochs = 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in tqdm(train_dataloader):
      # forward pass
      outputs = model(pixel_values=batch["pixel_values"].to(device),
                      input_boxes=batch["input_boxes"].to(device),
                      multimask_output=False)

      # compute loss
      predicted_masks = outputs.pred_masks.squeeze(1)
      ground_truth_masks = batch["ground_truth_mask"].float().to(device)
      loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

      # backward pass (compute gradients of parameters w.r.t. loss)
      optimizer.zero_grad()
      loss.backward()

      # optimize
      optimizer.step()
      epoch_losses.append(loss.item())

    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')

# Save the model's state dictionary to a file
torch.save(model.state_dict(), model_checkpoint)