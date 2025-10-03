from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences


from keras.applications.resnet50 import ResNet50

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


from keras.applications.resnet import preprocess_input

from keras.models import Model
from keras.models import load_model
import numpy as np
from image_preprocessing import image_transform
import random
import cv2

def create_croppings(numpy_array):
	# cropSize = 339
	# cellSize = 113
	# tileSize = 96
	cropSize = 225
	cellSize = 75
	tileSize = 64
	y_dim, x_dim = numpy_array.shape[:2]
	# Have the x & y coordinate of the crop
	crop_x = random.randrange(x_dim -  cropSize)
	crop_y = random.randrange(y_dim -  cropSize)
	# Select which image ordering we'll use from the maximum hamming set
	final_crops = np.zeros(
		( tileSize,  tileSize, 3, 9), dtype=np.float32)
	for row in range(3):
		for col in range(3):
			x_start = crop_x + col *  cellSize + \
				random.randrange( cellSize -  tileSize)
			y_start = crop_y + row *  cellSize + \
				random.randrange( cellSize -  tileSize)
			# Put the crop in the list of pieces randomly according to the
			# number picked
			final_crops[:, :, :,row * 3 + col]= numpy_array[y_start:y_start +  tileSize, x_start:x_start +  tileSize, :]
	x=np.transpose(final_crops,(3,0,1,2))
	for i,img in enumerate(x):
            # v=x[i,...]
            norm = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            x[i,...]=norm

	final_crops=x
	return final_crops

# extract features from each photo in the directory
def extract_features_full_architecture(directory):

	my_model = load_model("/home/student/Downloads/Semisupervised_Image_Classifier-master/SSL.hdf5")
	my_model = Model(inputs=my_model.inputs, outputs=my_model.layers[-2].output) # remove last dense layer (yeah, -2 means one before last)


	# extract features from each photo
	features = dict()
		# load an image from file
	filename = directory
	image = load_img(filename, target_size=(256, 256))
	image = img_to_array(image)
	image = create_croppings(image)
	# reshape data for the model
	image = image[:, np.newaxis,...]
	#change the first dimension of the array into list as expected by the model
	z=[]
	for i in image:
		z.append(i)       
	image=z
	# get features
	feature = my_model.predict(image, verbose=0)
	# get image id
	return feature
def extract_features_single_network(directory):


	my_model = load_model("/home/student/Downloads/Semisupervised_Image_Classifier-master/model_data/res50_384_3_65/model.hdf5")
	sourceModel = my_model.get_layer(index=9)  

	new_input = Input(shape=(384, 384, 3))
	targetModel = ResNet50(include_top=False, weights=None,
		input_tensor=new_input, pooling="avg") #this returns a blank Model 
	#coping the weights
	for l_tg,l_sr in zip(targetModel.layers, sourceModel.layers):
			wk0=l_sr.get_weights()
			l_tg.set_weights(wk0)
	targetModel.summary()


	# extract features from each photo
	features = dict()


	# load an image from file
	filename = directory
	image = load_img(filename, target_size=(384, 384))
	# convert the image pixels to a numpy array
	image = img_to_array(image)

	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	# # prepare the image for the VGG model
	image = preprocess_input(image)
	
	# get features
	feature = targetModel.predict(image, verbose=0)
	# get image id
	return feature


# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
		# convert probability to integer
		yhat = argmax(yhat)
		# map integer to word
		word = word_for_id(yhat, tokenizer)
		# stop if we cannot map the word
		if word is None:
			break
		# append as input for generating the next word
		in_text += ' ' + word
		# stop if we predict the end of the sequence
		if word == 'endseq':
			break
	return in_text
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)
def load_photo_features(filename, dataset):
	# load all features
	all_features = load(open(filename, 'rb'))
	# filter features
	features = {k: all_features[k] for k in dataset}
	return features
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))
# pre-define the max sequence length (from training)
max_length = 34
# load the model
filename = '/home/student/Downloads/Semisupervised_Image_Classifier-master/Flickr8k_text/Flickr_8k.testImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# descriptions
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descriptions: test=%d' % len(test_descriptions))
# photo features
test_features = load_photo_features('features_final_full.pkl', test)
print('Photos: test=%d' % len(test_features))

# load the model
model = load_model('/home/student/Downloads/Semisupervised_Image_Classifier-master/model_last-ep010-loss3.420-val_loss3.702.h5')
# load and prepare the photograph
for i in range(0,12) :
	photo = extract_features_full_architecture("example"+str(i)+".jpg")
	# generate description
	description = generate_desc(model, tokenizer, photo, max_length)
	print(str(i)+"   "+description)
