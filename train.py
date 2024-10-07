import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
from collections import OrderedDict

# Helper function for loading data
def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    # Define transforms for training and validation datasets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    # Load datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    
    return trainloader, validloader, train_dataset

# Build and train the model
def train_model(args):
    # Load data
    trainloader, validloader, train_dataset = load_data(args.data_directory)
    
    # Load a pre-trained network (e.g., VGG16)
    model = getattr(models, args.model)(pretrained=True)
    
    # Freeze parameters so we don’t backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Build custom classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, args.hidden_units)),
        ('relu', nn.ReLU()),
        ('dropout', nn.Dropout(0.2)),
        ('fc2', nn.Linear(args.hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Use GPU if requested and available
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training loop
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 40
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0
                
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train()
    
    # Save the model checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {
        'model': args.model,
        'hidden_units': args.hidden_units,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
    }
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
    
    print("Model training complete and saved!")

# Set up argument parser
def main():
    parser = argparse.ArgumentParser(description='Train an image classifier model.')
    parser.add_argument('--data_directory', type=str, required=True, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='./', help='Directory to save checkpoints')
    parser.add_argument('--model', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--hidden_units', type=int, default=512, help='Hidden units in classifier')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()
