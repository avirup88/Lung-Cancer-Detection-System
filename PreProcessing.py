#!/usr/bin/env python2
# -*- coding: utf-8 -*-

#%%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
from scipy import misc
from skimage import measure
import time
import cPickle
import cv2
from multiprocessing.dummy import Pool as ThreadPool 
import gzip
import tqdm
import statistics as stats
import statsmodels.api as sm
import peakutils


print ("Program Started....")

#%%
#1) Load image
# Load the scans in given folder path
def load_scan(path):
    try:
        slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    except BaseException as e:
        print str(e)
        
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
        
    for s in slices:
        try:
            s.SliceThickness = slice_thickness
        except BaseException as e:
            print str(e)
             
        
    return slices

#%%
#2) Convert pixels to HU    
#Function for converting pixel to Hounsfield Unit (HU)
def get_pixels_hu(slices):
    try:
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16), 
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)
    
        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0
        
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):
            
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)
                
            image[slice_number] += np.int16(intercept)
        
        return np.array(image, dtype=np.int16)
    except BaseException as e:
        print str(e)
        

#%%
#3) Resample image
#Function for resampling the image
def resample(image, scan, new_spacing=[1,1,1]):
    try:
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor
        
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
        
        return image, new_spacing
    except BaseException as e:
        print str(e)
        

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        biggest = vals[np.argmax(counts)]
    else:
        biggest = None
    return biggest

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

#%%
#5) Apply mask to resampled image
#Function for applying the mask 
def apply_mask(image, mask):
    masked_image = image * mask
    
    return masked_image

#%%    
#6) Select slices that contain nodules
#Function for applying the mask 
def tagSlicesNodules(image): 
    
    try:
    
        selected_slices = list()

        for i in range(len(image)):
                   
            sliceX = image[i]
            mode_val = stats.mode(sliceX.flatten())
            sliceX = sliceX[sliceX != mode_val]
            sliceX = sliceX[sliceX != 0.0]
            if (len(sliceX.flatten()) == 0):
                continue
            
            #Kernel Smoothing
            kde = sm.nonparametric.KDEUnivariate(sliceX.flatten())
            kde.fit(kernel='gau', bw='scott', fft=True)
            
            #Peak Detection for bimodal distribution
            indexes = peakutils.indexes(kde.density) 

            if (len(indexes)>1):
                cropped_image = image[i][min(np.where(image[i]!=0)[0]):max(np.where(image[i]!=0)[0]),min(np.where(image[i]!=0)[1]):max(np.where(image[i]!=0)[1])]
                rescaled_image = misc.imresize(cropped_image, size=(512,512))
                re_dimensioned_image = cv2.cvtColor(rescaled_image, cv2.COLOR_GRAY2BGR)
                selected_slices.append(re_dimensioned_image)
             
             
        print ("Number of Filtered Slices:" + str(len(selected_slices)))
        return selected_slices 
    except BaseException as e:
        print str(e)
        
MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

 
PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image       


def Pre_Processing(patient):
    try:
        print ("Current Patient: " + str(patient))
        df_ungrouped_results = pd.DataFrame(columns = ['id','slices'])
        patient_array = load_scan(INPUT_FOLDER + patient)
        patient_pixels = get_pixels_hu(patient_array)
        pix_resampled, spacing = resample(patient_pixels, patient_array, [1,1,1])
        mask = segment_lung_mask(pix_resampled, True)
        masked_image = apply_mask(pix_resampled, mask)
        normalized_image = normalize(masked_image)
        zero_centered_image = zero_center(normalized_image)
        preprocessed_slices = tagSlicesNodules(zero_centered_image)
        file = open(OUTPUT_FOLDER2 + 'patient_log.txt' ,'a')
        file.write("Patient_ID : " + patient + "    Number of Filtered Slices : " + str(len(preprocessed_slices)))
        file.write("\n")
        file.close()
        if (len(preprocessed_slices) > 0):
            df_ungrouped_results = df_ungrouped_results.append(pd.DataFrame({'id':patient,'slices':preprocessed_slices}),ignore_index=True)
            df_ungrouped_results = df_ungrouped_results.merge(labels, on = 'id', how='inner')
            if (len(df_ungrouped_results) > 0):
                cPickle.dump(df_ungrouped_results, gzip.GzipFile(OUTPUT_FOLDER + patient +'.p', 'wb'), -1)
                print ("Completed Patient: " + str(patient))
            else:
                print ("Skipped Patient: " + str(patient) + " due to lack of labels.")
                file = open(OUTPUT_FOLDER2 + 'skipped_no_label_patient_list.txt' ,'a')
                file.write("Patient_ID : " + patient)
                file.write("\n")
                file.close()        
        else:
            print ("Skipped Patient: " + str(patient) + " due to lack of visible tumor in the scans.")
            file = open(OUTPUT_FOLDER2 + 'skipped_no_tumor_patient_list.txt' ,'a')
            file.write("Patient_ID : " + patient)
            file.write("\n")
            file.close()            
    except BaseException as e:
        print str(e)
            
#%%
t0_1 = time.clock()
t0_2 = time.time()

#Selecting the paths

INPUT_FOLDER = raw_input("Enter the directory path of the input slices:")
label_filepath = raw_input("Enter the file of input labels:")
OUTPUT_FOLDER = raw_input("Enter the directory path of the output dataframe for selected slices:")
OUTPUT_FOLDER2 = raw_input("Enter the directory path of the log files")
Threads = raw_input("Enter the number of threads:")

labels = pd.read_csv(label_filepath)
patients_dir = os.listdir(INPUT_FOLDER)
patients = [x for x in patients_dir if not (x.startswith('.'))]
patients.sort()
print ("\nTotal Number of Patients in the original dataset: " + str(len(patients)) + "\n")
outdir_patients = len([name for name in os.listdir(OUTPUT_FOLDER) if name != ".DS_Store"])
print ("\nTotal Number of Patients processed: " + str(outdir_patients) + "\n")
if(outdir_patients > 0):
    patients = patients[outdir_patients+38:]
print ("\nTotal Number of Patients left to be processed: " + str(len(patients)) + "\n")

#Applying Multi-threading
pool = ThreadPool(int(Threads))
for _ in tqdm.tqdm(pool.imap_unordered(Pre_Processing, patients), total=len(patients)):
    pass
pool.close() 
pool.join()
    


print 'PROCESSING TIME'
print time.clock()-t0_1, 'seconds process time'
print time.time()-t0_2, 'seconds wall time'