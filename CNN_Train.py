
from __future__ import division, print_function, absolute_import
# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle
import numpy as np
import os
import pandas as pd
import gzip
import random

#Function to create the model architecture

def CNN_Model_Creation():
    # Make sure the data is normalized
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Create extra synthetic training data by flipping, rotating and blurring the
    # images on our data set.
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)
    
    # Define our network architecture:
    
    # Input is a 512x512 image with 3 color channels (red, green and blue)
    network = input_data(shape=[None, 512, 512,3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
                         
    # Step 1: Convolution
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
    
    # Wrap the network in a model object
    model = tflearn.DNN(network, tensorboard_verbose=0)

    return model

#Calling the Model Architecture
 
model = CNN_Model_Creation()


#Setting input parameters

INPUT_FOLDER = raw_input("Enter the directory path of the preprocessed slices:")
OUTPUT_FOLDER1 = raw_input("Enter the directory path of the output model file:")
OUTPUT_FOLDER2 = raw_input("Enter the directory path of the output test dataset files:")
label_filepath = raw_input("Enter the filename of input labels:")
labels = pd.read_csv(label_filepath)


labels_cancer = labels[labels['cancer']==1]
labels_no_cancer = labels[labels['cancer']==0]

labels_cancer_train_set = list(labels_cancer.id[:int(0.8*len(labels_cancer.cancer))])
labels_no_cancer_train_set = list(labels_no_cancer.id[:int(0.8*len(labels_no_cancer.cancer))])

labels_cancer_test_set = list(labels_cancer.id[int(0.8*len(labels_cancer.cancer)):])
labels_no_cancer_test_set = list(labels_no_cancer.id[int(0.8*len(labels_no_cancer.cancer)):])


pre_processed_files = os.listdir(INPUT_FOLDER)
pre_processed_files = [x for x in pre_processed_files if not (x.startswith('.'))]
random.shuffle(pre_processed_files)

pre_processed_file_ids = pd.Series([x[:-2] for x in pre_processed_files])

random.shuffle(pre_processed_file_ids)

pre_processed_train_set = pre_processed_files[:int(0.8*len(pre_processed_files))]

pre_processed_train_set = pre_processed_file_ids[pre_processed_file_ids.isin(labels_cancer_train_set)]
pre_processed_train_set = list(pre_processed_train_set.append(pre_processed_file_ids[pre_processed_file_ids.isin(labels_no_cancer_train_set)]))


pre_processed_test_set = pre_processed_file_ids[pre_processed_file_ids.isin(labels_cancer_test_set)]
pre_processed_test_set = list(pre_processed_test_set.append(pre_processed_file_ids[pre_processed_file_ids.isin(labels_no_cancer_test_set)]))



for curr_file in pre_processed_test_set: 
    # Load the data set
    source_path = gzip.GzipFile(INPUT_FOLDER + curr_file+".p", 'rb')    
    dataset_df = pickle.load(source_path)
    os.rename(INPUT_FOLDER + curr_file+".p", OUTPUT_FOLDER2+curr_file+".p")    

print ("Test Set Preparation Complete.")

print ("Starting Training Process...")

for curr_file in pre_processed_train_set[:50]:
    
    # Load the data set
    path = gzip.GzipFile(INPUT_FOLDER + curr_file+".p", 'rb')
    dataset_df = pickle.load(path)
    
    if (len(dataset_df.slices) == 0):
        print ("Ignoring patient :"+ str(curr_file) + " due to no slices.")
        continue

    if (len(dataset_df.slices) >= 70):
        X = np.array(list(dataset_df.slices[int(0.15*len(dataset_df.slices)):int(0.15*len(dataset_df.slices))+15])).astype(float)
        Y = np.array(list(dataset_df.cancer[int(0.15*len(dataset_df.slices)):int(0.15*len(dataset_df.slices))+15])).astype(float)
        inverted = np.where(Y==1,0,1)
        Y = np.vstack(zip(Y,inverted))
    else:
        X = np.array(list(dataset_df.slices)).astype(float)
        Y = np.array(list(dataset_df.cancer)).astype(float)
        inverted = np.where(Y==1,0,1)
        Y = np.vstack(zip(Y,inverted))

    
    # Shuffle the data
    X, Y = shuffle(X, Y)
        
    #Set the Batch Size
    batch = len(X)
             
    # Train it!
    model.fit(X, Y, n_epoch=1, shuffle=True,
              show_metric=True, batch_size=batch,
              snapshot_epoch=False,
              run_id='lung-classifier')
     

# Save model when training is complete to a file
model.save(OUTPUT_FOLDER1 + "lung_classifier.tfl")
print("Network trained and saved as lung_classifier.tfl!")


