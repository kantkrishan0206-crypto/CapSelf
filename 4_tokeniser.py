#-#-# Libs:

## Storage:
from pickle import dump # To store files of model (f. e. features) and dataset in more reliable way (for multiuser using)


## Keras:
from keras.preprocessing.text import Tokenizer

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

    # Save the tokenizer:
    dump(tokenizer, open('tokenizer.pkl', 'wb'))

#-#-#
