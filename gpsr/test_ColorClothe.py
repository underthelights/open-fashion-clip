from PIL import Image
from initCLIP import init_clip
from detectTopColor import detectTopColor
from detectTopClothe import detectTopClothe

# Initialize CLIP once
clip_model, preprocess, tokenizer, device = init_clip()

# Load and preprocess the image once
image_path = 'examples/shoes.jpg'
img = Image.open(image_path)
img = preprocess(img).to(device)

# Detect top color and clothing
top_color, top_color_prob = detectTopColor(img, clip_model, preprocess, tokenizer, device)
top_clothe, top_clothe_prob = detectTopClothe(img, clip_model, preprocess, tokenizer, device)

# Print the results
print(f"Top Color Detected: {top_color} with probability {top_color_prob}")
print(f"Top Clothing Detected: {top_clothe} with probability {top_clothe_prob}")
print("----")
print(f"{top_color} {top_clothe}")
