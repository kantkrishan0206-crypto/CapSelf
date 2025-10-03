from os import listdir
from pickle import dump
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet import preprocess_input
from keras.models import Model
from image_preprocessing import image_transform
from keras.models import Model,load_model
import numpy as np
from keras.layers import (Dense, Dropout, Concatenate, Input, Activation, Flatten, Conv2D,
                          MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, add)
import random
def create_croppings(numpy_array):
	cropSize = 339
	cellSize = 113
	tileSize = 96
# 	cropSize = 225
# 	cellSize = 75
# 	tileSize = 64
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
            mean, std = np.mean(x[i,...]), np.std(x[i,...])
            if std==0:
                continue #black image
            x[i,...]=(x[i,...]-mean)/std
	final_crops=x
	return final_crops
# extract features from each photo in the directory
def extract_features_single_network(directory):


	my_model = load_model("./model.hdf5") #pretrained model
	sourceModel = my_model.get_layer(index=9)  #ResNet index

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

	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
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
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features
def extract_features_full_architecture(directory):
	# load the model
	# # re-structure the model
	new_input = Input(shape=(256, 256, 3))
	my_model = load_model("./model.hdf5")
	my_model = Model(inputs=my_model.inputs, outputs=my_model.layers[-2].output) # remove last dense layer (yeah, -2 means one before last)
	# extract features from each photo
	features = dict()
	for name in listdir(directory):
		# load an image from file
		filename = directory + '/' + name
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
		image_id = name.split('.')[0]
		# store feature
		features[image_id] = feature
		print('>%s' % name)
	return features

# extract features from all images
directory = './Flicker8k_Dataset/'
features = extract_features_single_network(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))
