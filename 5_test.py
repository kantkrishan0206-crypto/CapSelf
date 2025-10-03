#-#-# Libs:

## Local directory files processing:
from os import listdir # To check files in a folder


## Storage:
from pickle import load


## Pre-processing:
import sys
import numpy
from numpy import argmax

# Keras pre-processing:
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

# PyTorch pre-processing:
import torchvision.transforms as transforms # Image transformations
from torch.autograd import Variable
from PIL import Image  # "Load image from file" function


## Models:
from pl_bolts.models.self_supervised import SimCLR # pre-trained SimCLR model (ResNet-50) by Pytorch Ligjhting framework


## PyTorch:
import torch
import torch.nn as nn

#-#-#




#-#-# Functions:

#### Image part functions:

#-# Extract features from an image:
def extract_features(filename):

    ## SimCLR model preparation:
    # Initialise pre-trained model with the pre-trained weights for SimCLR model:
    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt' 
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False) # SimCLR model initialisation
    simclr.freeze() # Freeze the model parameters for further model using
    #print(simclr) # Print the original model

    # Remove the projection head (which needed for training only):
    simclr_resnet50 = simclr.encoder

    # Numerate the model layers (to further remove the last one):
    children_counter = 0
    for n,c in simclr_resnet50.named_children():
        #print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # Remove the last fully-connected layer (needed for evaluation on ImageNet only):
    newmodel = torch.nn.Sequential(*(list(simclr_resnet50.children())[:-1])) 
    #print(newmodel) # Print the final model

    # Numerate the model layers (to further remove the last one):
    children_counter = 0
    for n,c in newmodel.named_children():
        #print("Children Counter: ",children_counter," Layer Name: ",n,)
        children_counter+=1

    # Set model to the evaluation state:
    newmodel.eval() 


    ## Image data pre-processing subfunctions:
    scaler = transforms.Resize((224, 224)) # Resize input image to 224x224 shape

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                    # Normilise image

    to_tensor = transforms.ToTensor() # Put normilised image in Tensor


	## Extract features from an image:
    # Load the image with Pillow library
    img = Image.open(filename)

    # Create a PyTorch variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

    # Get features from the last model's layer:
    feature_tens = newmodel(t_img) # Get features
    feature_nump = feature_tens.detach().cpu().numpy() # Convert them to the numpy array
    feature = feature_nump[:, :, 0, 0] # Reshape the array

    ## Return the feature vector:
    return feature
#-#


#-# Map an integer to a word:
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:

            return word

    return None
#-#



#-# Generate a description for an image:
def generate_desc(model, tokenizer, image, max_length):
    # Seed the generation process
    in_text = 'startseq'

    # Iterate over the whole length of the sequence:
    for i in range(max_length):
        # Integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]

        # Pad input
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict next word
        yhat = model.predict([image,sequence], verbose=0)

        # Convert probability to integer
        yhat = argmax(yhat)

        # Map integer to word
        word = word_for_id(yhat, tokenizer)

        # Stop if we cannot map the word
        if word is None:
            break

        # Append as input for generating the next word
        in_text += ' ' + word

        # Stop if we predict the end of the sequence
        if word == 'endseq':
            break

    return in_text
#-#

#-#-#




#-#-# Main:

if __name__ == '__main__': # !Don't forget to change the dataset paths!
    # Load the tokenizer
    tokenizer = load(open('Pre-trained/tokenizer.pkl', 'rb'))

    # Pre-define the max sequence length (we know it from training)
    max_length = 34

    # Load the model
    model = load_model('Pre-trained/model-ep011-loss2.804-val_loss3.415.h5') # Pre-trained model, change for your new model if any

    # Generate captions for random images:
    for i in range(12):
        # Load and prepare the image
        features = extract_features('Examples/example' + str(i) + '.jpg')

        # Generate description:
        print(i)
        description = generate_desc(model, tokenizer, features, max_length)
        print(description)

#-#-#
