# # Testing the network

# ## Import the required packages

# In[4]:

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import numpy as np
import pickle
import gzip


# ## Redefine the network
# Define the same network as initially created, along with all the preprocessing, blurring, and rotation of the data.

# In[5]:

# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)


# In[6]:

network = input_data(shape=[None, 512, 512, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 5, activation='relu')
    
network = conv_2d(network, 32, 3, activation='relu')
    
    # Step 2: Max pooling
network = max_pool_2d(network, 2)
    
    # Step 3: Convolution again
network = conv_2d(network, 64, 3, activation='relu')
    
    # Step 4: Convolution yet again
network = conv_2d(network, 64, 3, activation='relu')
    
    # Step 5: Max pooling again
network = max_pool_2d(network, 2)
    
network = conv_2d(network, 96, 2, activation='relu')
    
network = max_pool_2d(network, 2)
    
    # Step 6: Fully-connected 512 node neural network
network = fully_connected(network, 512, activation='relu')
    
    # Step 7: Dropout - throw away some data randomly during training to prevent over-fitting
network = dropout(network, 0.5)
    
    # Step 8: Fully-connected neural network with two outputs (0=isn't a cancer, 1=is a cancer) to make the final prediction
network = fully_connected(network, 2, activation='softmax')
    
#momentum = tflearn.optimizers.Momentum(learning_rate=0.05, momentum=0.7, lr_decay=0.5)    
    
    # Tell tflearn how we want to train the network
network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy')




# ## Reload the model

INPUT_FOLDER1 = raw_input("Enter the directory of the saved CNN model file:")
INPUT_TEST_FILE = raw_input("Enter the name of the test file:")

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load(INPUT_FOLDER1 + "lung_classifier.tfl")


# ## Test the model
# Test the restored model on some new image.


# Load the data set
path = gzip.GzipFile(INPUT_TEST_FILE, 'rb')
dataset_df = pickle.load(path)


#Predict values
prediction = model.predict(np.array(list(dataset_df.slices)).astype(float))

col1,col2 = zip(*prediction)
dataset_df['predicted_cancer'] = col1

final_group = dataset_df.groupby('id')
final_df = final_group.sum()

final_df['cancer'] = np.where(final_df.cancer > 0, 1, final_df.cancer)
final_df['predicted_cancer'] = np.where(final_df.predicted_cancer > 0, 1, final_df.predicted_cancer)

final_df.reset_index(level=0, inplace=True)

#Print the final output
print (final_df)
