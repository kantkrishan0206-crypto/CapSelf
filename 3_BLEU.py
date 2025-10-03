#-#-# Libs:

## Storage:
from pickle import load


## Pre-processing:
from numpy import argmax


## Keras:
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


## BLEU words corpus:
from nltk.translate.bleu_score import corpus_bleu

#-#-#




#-#-# Functions:


#-# Load text:
def load_text(filename):
    file = open(filename, 'r') # Open the file as read only
    text = file.read() # Read all text
    file.close() # Close the file

    return text
#-#



#-# Load a pre-defined list of image identifiers:
def load_set(filename):
    doc = load_text(filename)
    dataset = list()

    # Process line by line:
    for line in doc.split('\n'):
        # Skip empty lines
        if len(line) < 1:
            continue

        # Get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)

    return set(dataset)
#-# 



#-# Load clean descriptions into memory:
def load_clean_descriptions(filename, dataset):
    # Load document
    doc = load_text(filename)
    descriptions = dict()

    for line in doc.split('\n'):
        # Split line by white space
        tokens = line.split()

        # Split id from description
        image_id, image_desc = tokens[0], tokens[1:]

        # Skip images not in the set
        if image_id in dataset:
            # Create list
            if image_id not in descriptions:
                descriptions[image_id] = list()

            # Wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'

            # Store descriptions
            descriptions[image_id].append(desc)

    return descriptions
#-# 




#-# Load image features:
def load_image_features(filename, dataset):
    # Load all features
    all_features = load(open(filename, 'rb'))

    # Filter features
    features = {k: all_features[k] for k in dataset}
    
    return features
#-# 



#-# Covert a dictionary of clean descriptions to a list of descriptions:
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    
    return all_desc
#-# 


#-# Fit a tokenizer given caption descriptions:
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer
#-#



#-# Calculate the length of the description with the most words:
def max_length(descriptions):
    lines = to_lines(descriptions)

    return max(len(d.split()) for d in lines)
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



#-# Evaluate caption generation ability of the model:
def evaluate_model(model, descriptions, images, tokenizer, max_length):
    actual, predicted = list(), list()

    # Step over the whole dataset:
    for key, desc_list in descriptions.items():
        # Generate description
        yhat = generate_desc(model, tokenizer, images[key], max_length)

        # Store actual and predicted
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    # Calculate BLEU score:
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
#-#

#-#-#




#-#-# Main:

if __name__ == '__main__': # !Don't forget to change the dataset paths!
    ### Prepare tokenizer on train dataset:
    ## Load training images dataset (6K)
    filename = 'Flickr8k_text/Flickr_8k.trainImages.txt' # training dataset path
    #filename = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt' 
    train = load_set(filename) # Load dataset
    print('Dataset: %d' % len(train)) 


    ## Load training descriptions:
    train_descriptions = load_clean_descriptions('Pre-trained/descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))


    ## Load Tokenizer:
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    # Determine the maximum sequence length
    max_length = max_length(train_descriptions)
    print('Description Length: %d' % max_length)



    ### Evaluation:
    # Load test dataset
    filename = 'Flickr8k_text/Flickr_8k.devImages.txt' # Test dataset path
    #filename = '/home/dd/Documents/Datasets/Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
    test = load_set(filename) # Load dataset
    print('Dataset: %d' % len(test))


    ## Load test descriptions:
    test_descriptions = load_clean_descriptions('Pre-trained/descriptions.txt', test)
    print('Descriptions: test=%d' % len(test_descriptions))
    
    
    ## Load image features:
    test_features = load_image_features('Pre-trained/features.pkl', test)
    print('Images: test=%d' % len(test_features))

    ## Load the model:
    filename = 'Pre-trained/model-ep011-loss2.804-val_loss3.415.h5' # Pre-trained model, change for your new model if any
    model = load_model(filename) # Initialise the model

    print ("Starting BLEU score counting. This process may take 10-15 minutes.")

    ## Evaluate model:
    evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

#-#-#
