import json
import torch
import get_input_args


from PIL import Image
from torchvision import models
from torchvision import transforms


def main():
    """
        Image Classification Prediction
    """
    # load the cli args
    parser = get_input_args.get_args()
    args = parser.parse_args()

    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if args.use_gpu:
        device = torch.device("cuda:0")

    # load categories
    with open(args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # load model
    model = load_checkpoint(args.checkpoint)

    top_prob, top_classes = predict(args.image_path, model, args.top_k)

    label = top_classes[0]
    prob = top_prob[0]

   

    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")


def predict(image_path, model, topk=5):
        
    model.eval()
    
    model.cpu()
    
    #processing image
    image = process_image(image_path)
    
    #converting to tensor
    image = torch.from_numpy(image).type(torch.FloatTensor)
    
    #converting to tensor with size 1
    image = image.unsqueeze(0)
    
    output = model.forward(image)
    top_prob, top_labels = output.topk(topk)
    
    #exp
    top_prob = top_prob.exp()
    
    #invert the dictionary
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    classes = list()
    
    for label in top_labels.numpy()[0]:
        classes.append(class_to_idx_inv[label])
        
    return top_prob, classes


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    img = Image.open(image)
    
    # Resize with thumbnail
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    
     # Crop 
        
    width, height = img.size   # Get dimensions
    new_width = 224
    new_height= 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img=img.crop((left, top, right, bottom))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    img = (img - mean)/std
    
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

    print(f'Parameters\n---------------------------------')

    print(f'Image  : {args.image_path}')
    print(f'Model  : {args.checkpoint}')
    print(f'Device : {device}')


    print(f'FlowerCategory   : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')



