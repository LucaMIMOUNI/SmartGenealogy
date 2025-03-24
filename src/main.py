### Libraries
# For FCN - Fully Convolutional Networks
import os
import keras
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import ops
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# For the dataset
import skimage.io
from mnist import MNIST
import gzip
###

def __init__():
    NUM_CLASSES = 27
    INPUT_HEIGHT = 28
    INPUT_WIDTH = 28
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    EPOCHS = 20
    BATCH_SIZE = 32
    SHUFFLE = True
    
def display_images(image_path):
    img = skimage.io.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    #display_images("./data/FRAD35_80_10NUM_35130_00438_0002.jpg")
    # path = './training_dataset/gzip/emnist-letters-train-images-idx3-ubyte.gz'
    
    # f = gzip.open(path, 'r')
    # image_size = 28
    # num_images = 5
    # f.read(16)
    # buf = f.read(image_size * image_size * num_images)
    # data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    # data = data.reshape(num_images, image_size, image_size, 1)

    # image = np.asarray(data[0]).squeeze()
    # plt.imshow(image)
    # plt.show()
    print() 