import argparse
import train_utility
from train_utility import load_data, load_model, train_model, validate_model, save_model

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str, help='Directory of Flowers')
    parser.add_argument('--save_dir', type=str, default='save_directory', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg13','vgg16','vgg19'], help='Architectures available to be deployed')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')
    parser.add_argument('--hidden_units', type=int, default=1000, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', default=True, help='Runs network on GPU')
    
    return parser.parse_args()

def main():
    args = get_input_args()
    data_dir = args.data_directory
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu
    
    trainloader, validloader, testloader, class_to_idx = load_data(data_dir)
    model, criterion, optimizer, input_units = load_model(arch, hidden_units, learning_rate)
    train_model(model, epochs, criterion, optimizer, trainloader, validloader, gpu)
    test_loss, accuracy = validate_model(model, criterion, testloader, optimizer, gpu)
    save_model(arch, learning_rate, input_units, hidden_units, epochs, save_dir, class_to_idx, model, optimizer)
    
    print("Test Loss: {:.2f};".format(test_loss),
          "Test Accuracy: {:.2f}%".format(accuracy))
    
if __name__ == "__main__":
    main()