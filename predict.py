import torch
from torchvision import models
import argparse
import json
from PIL import Image
import numpy as np

# Load checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['model'])(pretrained=True)
    
    # Build classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, checkpoint['hidden_units'])),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(checkpoint['hidden_units'], 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Process input image
def process_image(image_path):
    img = Image.open(image_path)
    
    # Resize and crop the image
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    
    # Convert image to numpy array and normalize
    np_image = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image)

# Predict the class (or classes) of an image
def predict(image_path, model, topk, gpu):
    model.eval()
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    image = process_image(image_path).unsqueeze_(0).float()
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_class[0].cpu().numpy()]
    
    return top_p.cpu().numpy(), top_classes

# Set up argument parser
def main():
    parser = argparse.ArgumentParser(description='Predict the class of an input image.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category to name mapping JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    # Load the model
    model = load_checkpoint(args.checkpoint)
    
    # Predict the class
    probs, classes = predict(args.input, model, args.top_k, args.gpu)
    
    # If category names file provided, load and map class indices to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[cls] for cls in classes]
    
    print(f"Top {args.top_k} Classes: {classes}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
