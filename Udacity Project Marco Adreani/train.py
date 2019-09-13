
import json
import get_input_args
import torch

from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict

def main():
    parser = get_input_args.get_args()
    args = parser.parse_args()
    
    # load categories
    with open(args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # set output to the number of categories
    output_size = len(cat_to_name)
    print(f"Images are labeled with {output_size} categories.")
    
    #data directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # building dataloader
    

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 


    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(args.data_directory, transform=train_transforms) 
    
                                      
    #  Using the image datasets and the trainforms, define the dataloaders
     
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    
    
    
    # Building model
    model = models.__dict__[args.arch](pretrained=True)
    
    input_size = 0

    # Input size
    if args.arch.startswith("vgg"):
        input_size = nn_model.classifier[0].in_features
    
    if args.arch.startwith("densenet"):
        input_size = 1024
        
    # Prevent back propagation on parameters
    for param in nn_model.parameters():
        param.requires_grad = False
        
    # size of hidden units
    hidden_sizes = args.hidden_units
    
    for i in range(len(hidden_sizes) - 1):
       OrderedDict(['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
       OrderedDict(['relu' + str(i + 1)] = nn.ReLU())
       OrderedDict(['dropout' + str(i + 1)] = nn.Dropout(0.5))

    OrderedDict(['output'] = nn.Linear(hidden_sizes[i + 1], output_size))
    OrderedDict(['softmax'] = nn.LogSoftmax(dim=1))
    
    classifier = nn.Sequential(OrderedDict())

   
    model.classifier = classifier

    # Start clean by setting gradients of all parameters to zero.
    nn_model.zero_grad()

    # criterion NLLLoss
    criterion = nn.NLLLoss()
    
    # optimizer : Adam
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learnin_grate)
    
    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Using CPU.")
        
    model = model.to(device)
    
    #Training
    
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(args.epochs):
        for inputs, labels in trainloader:
            steps += 1
       
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                     for inputs, labels in testloader:
                         inputs, labels = inputs.to(device), labels.to(device)
                         logps = model.forward(inputs)
                         batch_loss = criterion(logps, labels)
                    
                         test_loss += batch_loss.item()
                    
                         # Calculate accuracy
                         ps = torch.exp(logps)
                         top_p, top_class = ps.topk(1, dim=1)
                         equals = top_class == labels.view(*top_class.shape)
                         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                       f"Train loss: {running_loss/print_every:.3f}.. "
                       f"Test loss: {test_loss/len(testloader):.3f}.. "
                       f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
            
            
     # Saving model
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'epoch': args.epochs,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}

     torch.save(model_state, args.save_dir)





                               
        
        