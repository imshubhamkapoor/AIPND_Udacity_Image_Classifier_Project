import argparse
import json
import torch
import predict_utility
from predict_utility import load_checkpoint, predict

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='Image file path')
    parser.add_argument('checkpoint', type=str, help='Saved Checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File containing categories of flowers')
    parser.add_argument('--gpu', action='store_true', default=True, help='Runs network on GPU')
    
    return parser.parse_args()

def main():
    args = get_input_args()
    image = args.image
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu

    model = load_checkpoint(checkpoint)
        
    with open(category_names, 'r') as f:
        categories = json.load(f)
        
    top_p, top_class = predict(image, model, top_k, gpu)
    
    for i in range(0, len(top_class)):
        print('The most likely class is: {}; probability: {}'.format(categories[top_class[i]], top_p[i]))
        
if __name__ == "__main__":
    main()