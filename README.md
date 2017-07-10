# Lung-Cancer-Detection-System

This project proposes a method which tries to improve on the lung cancer detection system by proper segmentation of lung nodules on different slices of the CT scans and then tries to apply deep learning methodology like Convolution Neural Networks (CNN) using TensorFlow framework on those segmented scan slices and discards the unnecessary information in order to narrow down the relevant slices and predict whether the patient has lung cancer in the final layer of the fully connected network. The method is divided into 2 major sub-sections – Pre-processing of the CT scans into a resampled and rescaled 3D segmentation of lungs, the preprocessed image is then fed into a two tier ConvNet which transforms from a coarse to fine cascade framework that leads to reduction of the false positive rate from the Tier I to Tier II of the analysis. Initially, the segmented lung image is fed into the Tier I ConvNet which identifies the necessary region of interest (ROI) in order to remove those CT scan slices that doesn’t concern our analysis. This improves the accuracy as well as the computational performance. The selective slice reduction of CT scans acts as input to train the Tier II deep convolution neural network (ConvNet). This second tier behaves as a highly selective process to reject difficult false positives while preserving high sensitivities.
