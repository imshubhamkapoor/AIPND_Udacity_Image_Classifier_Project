# AIPND_Udacity_Image_Classifier_Project
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories

The project is broken down into multiple steps:
- Load and preprocess the image dataset
- Train the image classifier on your dataset
- Use the trained classifier to predict image content

Firstly, each part will be implemented in Python. When you've completed this project, you'll have an application that can be trained on any set of labelled images. Here your network will be learning about flowers and will end up as a command line application.

## Files Description

- **Image Classifier Project.ipynb** It is used to build the model using the jupyter notebook. It can be used independently to see how the model works.
- **cat_to_name.json** It is used in ipynb and py files to map flower number to flower names.
- **train_utility.py** It is used by train.py to enable the parameter function.
- **train.py** It will train a new network on a dataset and save the model as a checkpoint.
- **predict_utility.py** It is used by predict.py to enable the parameter function.
- **predict.py** It uses a trained network to predict the class for an input image.

## Installation
The Code is written in Python 3.6.5 . If you don't have Python installed you can find it [here](https://www.python.org/downloads/).
If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip.

To install pip, run in the command Line:

`python -m ensurepip -- default-pip`

To upgrade Python, run in the command Line:

`pip install python -- upgrade`

Additional Packages that are required are: Numpy, Pandas, MatplotLib, Pytorch, PIL and json. You can donwload them using pip

`pip install numpy pandas matplotlib pil`

In order to intall Pytorch head over to the [Pytorch](https://pytorch.org/get-started/locally/) website and follow the instructions given.

## Command Line Application
- Train a new network on a data set with train.py. Prints out training loss, validation loss, and validation accuracy as the network trains and stores the trained model in a directory
  * Basic usage: python train.py data_directory
  * Options:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture (available: vgg13, vgg16, vgg19): python train.py data_dir --arch "vgg16"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 1000 --epochs 3
    - Use GPU for training: python train.py data_dir --gpu

- Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
  * Basic usage: python predict.py /path/to/image checkpoint
  * Options:
    - Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
    - Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
    - Use GPU for inference: python predict.py input checkpoint --gpu
    
## Data
The data used specifically for this assignment are a flower database(.json file). It is not provided in the repository as it's larger than what github allows.
The data need to comprised of 3 folders:
- test
- train
- valid

Generally the proportions should be 70% training 10% validate and 20% test.
Inside the train, test and valid folders there should be folders bearing a specific number which corresponds to a specific category, clarified in the json file. For example, if we have the image x.jpg and it is a lotus it could be in a path like this /test/5/x.jpg and json file would be like this {...5:"lotus",...}.

## GPU/CPU

As this project uses deep CNNs, for training of network you need to use a GPU. However after training you can always use normal CPU for the prediction phase.

## License
[MIT License]
