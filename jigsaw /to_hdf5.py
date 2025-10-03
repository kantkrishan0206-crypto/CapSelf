from time import strftime, localtime, time
import random
import h5py
import numpy as np
from PIL import Image
import glob
import random
import numpy as np
from keras.preprocessing.image import img_to_array
import cv2
# Make size slightly larger than the crop size in order to add a little
# more variation
SIZE = (256, 256)
TEST_RUN = False
TEST_SUBSET_DATA = False


directory = "./unlabeled2017/"

files = glob.glob(directory + "*.jpg")

# Shuffle the data
random.shuffle(files)
# Split data 70% train, 15% validation, 15% test
files_dict = {}

# Using Welford method for online calculation

# Create the HDF5 output file

files_dict["train_img"] = files[:int(0.85 * len(files))]
files_dict["val_img"] = files[int(0.85 * len(files)):int(0.95 * len(files))]
files_dict["test_img"] = files[int(0.95 * len(files)):]
print("Length of files array: {}".format(len(files)))
hdf5_output = h5py.File("./COCO_2017_unlabeled.hdf5", mode='w')
hdf5_output.create_dataset(
"train_img", (len(files_dict["train_img"])*3, *SIZE, 3), np.uint8)
hdf5_output.create_dataset(
"val_img", (len(files_dict["val_img"])*3, *SIZE, 3), np.uint8)
hdf5_output.create_dataset(
"test_img", (len(files_dict["test_img"])*3, *SIZE, 3), np.uint8)


start_time = time()
small_start = start_time

for img_type, img_list in files_dict.items():
    for index, fileName in enumerate(img_list):
        im = Image.open(fileName)
        # convert black and white images to 3 channels
        if (im.mode != 'RGB'):
            gray = img_to_array(im)
            x=gray.shape
            img2 = np.zeros((x[0],x[1],3))
            img2[:,:,0] = gray[:,:,0]
            img2[:,:,1] = gray[:,:,0]
            img2[:,:,2] = gray[:,:,0]
            im=Image.fromarray(img2.astype(np.uint8))
            gray2 = img_to_array(im)
            # If its taller than it is wide, crop first
        if (im.size[1] > im.size[0]):
            crop_shift = random.randrange(im.size[1] - im.size[0])
            im = im.crop(
                (0, crop_shift, im.size[0], im.size[0] + crop_shift))
        elif (im.size[0] > im.size[1]):
            crop_shift = random.randrange(im.size[0] - im.size[1])
            im = im.crop(
                (crop_shift, 0, im.size[1] + crop_shift, im.size[1]))
        im = im.resize(SIZE, resample=Image.LANCZOS)
        numpy_image = np.array(im, dtype=np.uint8)
        # Save the image to the HDF5 output file
        hdf5_output[img_type][index, ...] = numpy_image
        if index % 1000 == 0 and index > 0:
            small_end = time()
            print("Saved {} {}s to hdf5 file in {} seconds".format(
                index, img_type, small_end - small_start))
            small_start = time()



end_time = time()
print("Elapsed time: {} seconds".format(end_time - start_time))
