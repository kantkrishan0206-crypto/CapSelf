import h5py
import random
import time
import numpy as np
from image_preprocessing import image_transform
import itertools
import warnings
import threading
from PIL import Image

class DataGenerator:
    """
    Class for a generator that reads in data from the HDF5 file, one batch at
    a time, converts it into the jigsaw, and then returns the data
    """

    def __init__(self, maxHammingSet, image_size=256,xDim=64, yDim=64, numChannels=3,
                 numCrops=9, batchSize=32):
        """
        meanTensor - rank 3 tensor of the mean for each pixel for all colour channels, used to normalize the data
        stdTensor - rank 3 tensor of the std for each pixel for all colour channels, used to normalize the data
        maxHammingSet - a
        """
        self.xDim = xDim
        self.yDim = yDim
        self.numChannels = numChannels
        self.numCrops = numCrops
        self.batchSize = batchSize
        self.image_size=image_size
        self.maxHammingSet = np.array(maxHammingSet, dtype=np.uint8)
        # Determine how many possible jigsaw puzzle arrangements there are
        self.numJigsawTypes = self.maxHammingSet.shape[0]
        # Use default options for JigsawCreator
        self.jigsawCreator = image_transform.JigsawCreator(
            maxHammingSet=maxHammingSet)

    def __data_generation_normalize(self, dataset, batchIndex):
        """                              
        Internal method used to help generate data, used when
        dataset - an HDF5 dataset (either train or validation)
        """
        # Determine which jigsaw permutation to use
        x = np.empty((self.batchSize, self.image_size, self.image_size, self.numChannels),
                     dtype=np.float32)

        x = dataset[batchIndex * self.batchSize:(batchIndex + 1) * self.batchSize, ...].astype(np.float32)
        X = np.empty((self.batchSize, self.xDim, self.yDim,
                      self.numCrops), dtype=np.float32)
        y = np.empty(self.batchSize)
        # Python list of 4D numpy tensors for each channel
        X = [np.empty((self.batchSize, self.xDim, self.yDim,
                       self.numChannels), np.float32) for _ in range(self.numCrops)]
        #  pdb.set_trace()
        for image_num in range(self.batchSize):
            # Transform the image into its nine croppings
            single_image, y[image_num] = self.jigsawCreator.create_croppings(
                x[image_num])
            for image_location in range(self.numCrops):
                X[image_location][image_num, :, :,
                                  :] = single_image[:, :, :, image_location]
       

        y=self.sparsify(y)
        return X, y

    def sparsify(self, y):
        """
        Returns labels in binary NumPy array
        """
        return np.array([[1 if y[i] == j else 0 for j in range(self.numJigsawTypes)]
                         for i in range(y.shape[0])])

    #  @threadsafe_generator
    def generate(self, dataset):
        """
        dataset - an HDF5 dataset (either train or validation)
        """
        numBatches = dataset.shape[0] // self.batchSize
        batchIndex = 0
        while True:
             # Load data
            X, y = self.__data_generation_normalize(dataset, batchIndex)
            batchIndex += 1  # Increment the batch index
            if batchIndex == numBatches:
                # so that batchIndex wraps back and loop goes on indefinitely
                batchIndex = 0
            yield X, y
