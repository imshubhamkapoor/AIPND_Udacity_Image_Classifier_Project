import torch
from torchvision import models
from PIL import Image
import numpy as np
from torchvision import transforms
import train_utility
from train_utility import load_model

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['model_name']
    hidden_units = checkpoint['hidden_layer']
    learning_rate = checkpoint['learning_rate']
    model, criterion, optimizer, input_units = load_model(arch, hidden_units, learning_rate)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    optimizer = checkpoint['optimizer']
    epochs = checkpoint['epochs']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    adjust = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])
    
    pil_image = adjust(pil_image)
    np_image = np.array(pil_image)
    
    return np_image 

def predict(image_path, model, top_k, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.eval()
    img = process_image(image_path)
    if gpu and torch.cuda.is_available():
        print('Running on GPU')
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        model.cuda()
    elif gpu and torch.cuda.is_available()==False:
        print('GPU not found therefore running on CPU')
        img = torch.from_numpy(img).type(torch.FloatTensor)
        model.cpu()
    else:
        print('Running on CPU')
        img = torch.from_numpy(img).type(torch.FloatTensor)
        model.cpu()
    
    with torch.no_grad():
        img = img.unsqueeze_(0)
        logps = model(img)
        linearps = torch.exp(logps)
        top_p, top_idx = linearps.topk(top_k)
        idx_to_class = {x: y for y, x in model.class_to_idx.items()}
        top_idx = top_idx.data.cpu().numpy().squeeze()
        top_class = [idx_to_class[each] for each in top_idx]
        top_p = np.array(top_p)[0]
    
    return top_p, top_class