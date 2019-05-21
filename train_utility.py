import torch
import os
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.Resize(255),
                                           transforms.RandomCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    #Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    class_to_idx = train_datasets.class_to_idx
    
    return trainloader, validloader, testloader, class_to_idx

def load_model(arch, hidden_units, learning_rate):
    if arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        input_units = model.classifier[0].in_features
    elif arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        input_units = model.classifier[0].in_features
    elif arch == 'vgg19':
        model = models.vgg19(pretrained = True)
        input_units = model.classifier[0].in_features
    else:
        raise Exception('Model not available')
    # Building the network and defining classifier
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(nn.Linear(input_units, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(p=0.4),
                               nn.Linear(hidden_units, 102),
                               nn.LogSoftmax(dim=1))
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer, input_units

def validation(model, dataloader, optimizer, criterion, gpu=True):
    
    if gpu and torch.cuda.is_available():
        model.cuda()
    
    data_loss = 0
    accuracy = 0
    for images, labels in dataloader:
        
        if gpu and torch.cuda.is_available():
            images, labels = images.to('cuda'), labels.to('cuda')
            
        optimizer.zero_grad()
        logps = model(images)
        loss = criterion(logps, labels)
        data_loss += loss.item()
        
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
        
    return data_loss, accuracy

def train_model(model, epochs, criterion, optimizer, trainloader, validloader, gpu):
    steps = 0
    print_every = 8
    
    if gpu and torch.cuda.is_available():
        print('Training on GPU')
        model.cuda()
    elif gpu and torch.cuda.is_available()==False:
        print('GPU not found therefore training on CPU')
        model.cpu()
    else:
        print('Training on CPU')
        model.cpu()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for i,(images, labels) in enumerate(trainloader):
            steps += 1
            if gpu and torch.cuda.is_available():
                images, labels = images.to('cuda'), labels.to('cuda')
            
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, optimizer, criterion, gpu=True)
                
                    print("Epoch: {}/{}; ".format(epoch+1, epochs),
                          "Training Loss: {:.3f}; ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}; ".format(valid_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}%".format(accuracy*100/len(validloader)))
            
                running_loss = 0
                
def validate_model(model, criterion, testloader, optimizer, gpu):
    model.eval()
    
    if gpu and torch.cuda.is_available():
        print('Testing on GPU')
        model.cuda()
    elif gpu and torch.cuda.is_available()==False:
        print('GPU not found therefore testing on CPU')
        model.cpu()
    else:
        print('Testing on CPU')
        model.cpu()
        
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, optimizer, criterion, gpu=True)
        
        test_loss = test_loss/len(testloader)
        accuracy = accuracy*100/len(testloader)
        
        return test_loss, accuracy
    
def save_model(arch, learning_rate, input_units, hidden_units, epochs, save_dir, class_to_idx, model, optimizer):    
    checkpoint_param = {'model_name': arch,
                        'learning_rate': learning_rate,
                        'input_layer': input_units,
                        'hidden_layer': hidden_units,
                        'output_layer': 102,
                        'epochs': epochs,
                        'state_dict': model.state_dict(),
                        'class_to_idx': class_to_idx,
                        'classifier': model.classifier,
                        'optimizer': optimizer.state_dict}
    
    if os.path.exists(save_dir):
        checkpoint = save_dir +'/'+arch+'_checkpoint.pth'
    else:
        os.makedirs(save_dir)
        checkpoint = save_dir +'/'+arch+'_checkpoint.pth'

    torch.save(checkpoint_param, checkpoint)