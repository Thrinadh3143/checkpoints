import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Set environment variable for MPS fallback on unsupported operations (if using an MPS device)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Select the computation device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    print("CUDA and MPS are not available. Using CPU instead.")
print(f"Using device: {device}")

# Seed for reproducibility
np.random.seed(3)

# Display annotations
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)

# Load and display the image
image_path = '/Users/thrinadh_tellagorla/Downloads/car.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found at {image_path}")

image = Image.open(image_path).convert("RGB")
image = np.array(image)

plt.figure(figsize=(20, 20))
plt.imshow(image)
plt.axis('off')
plt.show()

# Load the SAM model
sam2_checkpoint = "/Users/thrinadh_tellagorla/sam2/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# Initialize the automatic mask generator and generate masks
mask_generator = SAM2AutomaticMaskGenerator(sam2)
masks = mask_generator.generate(image)
print(f"Number of masks generated: {len(masks)}")
print(f"Keys in each mask dictionary: {masks[0].keys()}")

# Define the path to save the output in the Downloads folder
output_path = '/Users/thrinadh_tellagorla/Downloads/output_segmentation.png'

# Generate and display the masks, then save to Downloads
plt.figure(figsize=(20, 20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Output image saved to {output_path}")
