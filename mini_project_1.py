import scipy.io
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# used to flatten array into 1D list -> img_label_extractor
from itertools import chain 

from numpy.linalg import inv
from numpy.linalg import matrix_rank

# used to find repeat values and indices 
from collections import defaultdict
mat = scipy.io.loadmat('mnist.mat')

np.set_printoptions(threshold=100)  # Set threshold to infinity to print all elements
def prediction_calc(arr):
    
    row,columns = np.shape(arr)
    print(row)
    print(columns)
    res = np.sum(arr == arr.max(axis=1, keepdims=True), axis=1) > 1
    prediction = np.zeros((10000,1))
    for y in range(row):
        np.put(prediction,[y][0],(np.argmax(arr[y][:])))
    return prediction
def visualize(img_array: np.array, index: int) -> None:
    '''
    Purpose: To visualize img data for manual verification of training, and fun 
    
    Parameters
    ----------
    index: int 
        Index of requested image data 
    img_array: np.array
        Normalized pixel information 
    
    Returns 
    -------
    Outputs requested image, no return value 
    
    
    
    '''
    
    # Extracts requested row from norm_trainX, reshapes into 28x28 pixel image and displays
    # Difference between using the normalized dataset and non-normalized is nothing...? 
    number = img_array[index].reshape((28, 28)) 
    plt.imshow(number)  
    plt.axis('off')  
    plt.show()
    
    
def info(values, printr):
    
    # The debugger takes forever to load this is way quicker #dontjudgemepls

    print(type(values))
    print(np.shape(values))
    if printr == 1:
        print(values)
    print('-------------')
def zero_remover_norm(img_array: np.array) -> np.array:
    '''
    Purpose: Process train/testX datasets, removes 0 columns, normalizes, and adds ones column 
    
    Parameters
    ----------
    img_array: np.array 
        Represents the image data  
        
    Returns 
    ----------
    norm_nonzero_img: np.array
        Image data with the 0 columns removed and normalized (all indices /255)
    '''
    
    # Checks for where columns are all 0's and removes them 
    # Input explanation as to why this is necessary within lab report 
    idx = np.argwhere(np.all(img_array[..., :] == 0, axis=0))
    img_array_zero = np.delete(img_array, idx, axis=1)
    
    # Normalizes image data set 
    # Input explanation as to why this is necessary within lab report 
    norm_nonzero_img = img_array_zero/255
    
    r,c = np.shape(norm_nonzero_img)
    ones_column = np.ones((r,1))
    trainX_norm_nonzero_ones = np.hstack((norm_nonzero_img,ones_column))

    return trainX_norm_nonzero_ones, idx
def sign(trainY: np.array, label_rqst: int) -> np.array:
    '''
    Binary classifier on trainY dataset, alters to 1 or -1 if index matches label_rqst
    
    
    Parameters
    ---------
    trainY: np.array 
        Y label training dataset 
    label_rqst: int
        requested label 
        
        
    Returns 
    -------
    trainY_bin: np.array
        trainY set processed into a set of 1's or -1's dependent on if matching label_rqst
    ''' 
    
 
    trainY_bin = np.where(trainY == label_rqst, 1, -1)
    
    return trainY_bin.T    
def img_label_extractor(norm_nonzero_img: np.array, train_label: np.array, label_rqst1: int, label_rqst2:int, original_img) -> np.array:
    '''
    Purpose: Extracts the img_data corresponding to the desired label 
    
    Parameters
    ----------
    norm_nonzero_img: np.array 
        Normalized and zeroed columns removed dataset 
    label_array: np.array
        True labels of our test image dataset 
        
        
    Returns 
    ----------
    labeled_img_data: np.array
        Extracted img data that is only of corresponding requested label  
        
    General process
    ----------
    Check requested label 
    - Get positions of labels in y matrix that matches with label_rqst
    - Extract corresponding indices from normalized trainX set 
    ''' 
    
    # Checks for requested label across train_label array and stores into idx 
    # Need to convert idx from np.array into flattened 1D list to correctly extract 
    # corresponding image data 
    
# Checks train_label indice values that match label_rqstN and stores the indice - slices indices from norm_nonzero_img
    #-------------------------------------
    idx1 = np.argwhere(np.all(train_label[..., :] == label_rqst1, axis=0))
    idx2 = np.argwhere(np.all(train_label[..., :] == label_rqst2, axis=0))
    idx1_flat = list(chain.from_iterable(idx1.tolist()))
    idx2_flat = list(chain.from_iterable(idx2.tolist()))
    indx_net = sorted(idx1_flat + idx2_flat)
    rqst_img_both = norm_nonzero_img[indx_net]
    #-------------------------------------
    
# Finds elements that aren't labelrqst1 or 2 and removes them 
    idx1_r = np.argwhere(np.all(train_label[..., :] != label_rqst1,axis = 0 ))
    idx2_r = np.argwhere(np.all(train_label[..., :] != label_rqst2,axis = 0 ))
    idx1_rflat = list(chain.from_iterable(idx1_r.tolist()))
    idx2_rflat = list(chain.from_iterable(idx2_r.tolist()))


    idx_r_common = [common for common in idx1_rflat if common in idx2_rflat]

    trainY_OVO = np.delete(train_label,idx_r_common)
    return rqst_img_both, trainY_OVO

def normal_eqn_solver(trainX_norm_nonzero_ones: np.array, trainY_bin: np.array) -> np.array:
    '''
    Purpose: Computes the solution to the normal equation 
    
    Parameters
    ----------
    img_array_zero: np.array 
        Image data with the zero columns removed 
    label_array: np.array
        True labels of our test image dataset 
        
        
    Returns 
    ----------
    weights: np.array
        Weighted matrix that contains the solution to our normal equation 
        
    theta = inv(X^T * X) * X^T * y
    ''' 
    
    
    
    # Add ones to the end of each row in order to account for bias factor in normal equation
    
    XTX = np.dot(np.transpose(trainX_norm_nonzero_ones),trainX_norm_nonzero_ones)
    x = np.shape(XTX) # rank of XTX 
   
    ## ADD IN SECTION ABOUT HOW THEY ARE NOT EQUAL!!!



    weights = np.dot(np.linalg.pinv(trainX_norm_nonzero_ones) , trainY_bin)
    return weights 


    ## not full rank therefore no closed form solution -> psuedo inverse solution needed 
    # we removed the zero columns in order to try and get the XTX inverse to be computable, but it still wasnt. However, we have less parameters so its still 
    # valid to remove the zero columns 
    
    # determinant of this X^T * X is 0 - singular matrix error 
def binary_classifier(weights: np.array, X_N_test_ones: np.array) -> np.array:
    '''
    Purpose: Create a one-versus-all classifier 
    
    Parameters
    ----------
    norm_nonzero_img: np.array
        normalized, nonzero columns image data 
        
    Returns 
    ----------
    confidence: np.array
        confidence level on each of the labels 

    ''' 
    confidence = np.dot(X_N_test_ones, weights)
    classifier_labels = np.sign(confidence)
    return classifier_labels, confidence
    #max_xy = np.where(max_ == a.max() )
    #isualize(norm_nonzero_img, 5000)

    #results = np.square(trainY-(np.dot(np.transpose(weights),norm_nonzero_img)))
def mc_one_vs_all_classifier(trainY: np.array, weights: np.array, X_N_test_ones: np.array):
    '''
    Multiclass classifier model 
    
    
    Parameters
    ---------
    trainY: np.array 
        Y label training dataset 
    weights: array
        weighted matrix of all weights of one vs all classifiers  
        
        
    Returns 
    -------
    mc_ova_labels: np.array
        prediction label array of ingested image dataset 
    ''' 
    
    confidence= np.dot(X_N_test_ones, weights)


    # Finding the indices of the maximum values in each row
    max_indices = np.argmax(confidence, axis=1)
    mc_ova_labels = []
    mc_ova_labels.append(np.argmax(confidence,axis=1))

    return mc_ova_labels

def mc_one_vs_one_classifier(labels: np.array,confidence,X_N_test_ones, pairs):
    print('one v one called')
    r, c = np.shape(labels)
    
    tally = np.zeros((10000,10)) 
    
    row_vote = np.zeros((10000,1))
    
    for row in range(r): # replace with r when running full classifier 
        for label_val_index in range(c):
            #print(f'Label val index: {label_val_index}')
            print(row)
            label_val_bin = labels[row][label_val_index] 
            onevone_pair = pairs[label_val_index] # should be a specific pair [0,9]
            if label_val_bin == 1:  
                tally[row][onevone_pair[0]] =  tally[row][onevone_pair[0]] + 1 
            elif label_val_bin == -1:
                tally[row][onevone_pair[1]] =  tally[row][onevone_pair[1]] + 1 
            else: 
                print("ERROR IN LABEL BIN VECTOR")
    print(tally[:10][:10])
    prediction_vec = prediction_calc(tally)   
    print(np.shape(prediction_vec))
    print(f'prediction values of first 10 elem : {prediction_vec[:10][0]}')          
    return prediction_vec

def confusion_matrix(testY, classifier_labels) -> int:
    #print(testY.T[5][0])
    '''
    Confusion Matrix calculation  
    
    
    Parameters
    ---------
    trainY: np.array 
        Y label training dataset 
    classifier labels: array
        the output of the classifier and what it labeled each index 
        as   
        
        
    Returns 
    -------
    err_perc: int
        The percentage error  
    ''' 
        

    
    info(testY,0)
    info(classifier_labels,0)
    error = 0 

    # Error Calculation 
    # -------------------------------------------
    c,r = np.shape(classifier_labels)
    # Ensure c is ranging over the correct value 
    for x in range(c):
        if testY[x] != classifier_labels[x]:
            error = error+1 
    err_perc = error/c
    print("Error Percent: " + str(err_perc) + " Error Norm: " + str(error))
    
    
    return err_perc
    # -------------------------------------------
    
    '''
    
    
    conf_mat = np.zeros(4)
    # 0 = TP 
    # 1 = FP 
    # 2 = FN 
    # 3 = TN 
    for y in range(c): 
        if testY[y] == classifier_labels[y]:
            conf_mat[0] = conf_mat[0] + 1
        elif test[y] != classifier_labels[y]:
            conf_mat[3] = conf_mat[0] + 1 
    '''
        
    
    

def main():

# Setup to compute the weighted matrix solution to the normal equation 
    # --------------------------------------------------
    trainX = mat['trainX']
    trainY = mat['trainY']

    X_N_train_ones, rmved_c = zero_remover_norm(trainX)
    # --------------------------------------------------

# testX manipulation - normalizes, removes same 0 columns as trainX, and adds 1 column - should recognifugre to be 
    # --------------------------------------------------
    testX_normal = mat['testX']/255
    testY = mat['testY']
    X_N_test = np.delete(testX_normal, rmved_c, axis=1)
    r = 10000
    ones_column = np.ones((r,1))
    X_N_test_ones = np.hstack((X_N_test,ones_column))
    # X_N = Normalized, Zero Columns removed, and ones column tacked on the end 
    # --------------------------------------------------


# Binary one versus all classifier test - 9 versus all 
    # -------------------------------------------------
    '''
    binary_class = 9
    trainY_bin = sign(trainY, binary_class)

    bin_weight = normal_eqn_solver(X_N_train_ones,trainY_bin)
    testY_bin = sign(testY,binary_class)
    bin_classifier_predictions, confidence = binary_classifier(bin_weight,X_N_test_ones)   

    confusion_matrix(testY_bin,bin_classifier_predictions)
    '''
    # --------------------------------------------------


# Multi-class one-versus all classifier 
    # --------------------------------------------------
    
    
    '''
    weight_list = []
    for x in range(10):
        trainY_bin = sign(trainY, x)
        weights = normal_eqn_solver(X_N_train_ones, trainY_bin)
        weight_list.append(weights)
        print(x) # Indicates how many loops till completion 
    weight_matrix = np.hstack(weight_list)
    mc_classifier_predictions = mc_one_vs_all_classifier(trainY, weight_matrix, X_N_test_ones)
    MC_error = confusion_matrix(np.transpose(testY), np.transpose(mc_classifier_predictions))
    '''
    # --------------------------------------------------
  
  
# Multi-class one versus one classifier 
    '''
    Thought process 
    - Conduct a single one vs one classifier first, 0 vs 1 
        - Data set up required -> extract 0 & 1's from trainX and trainY 
    
    '''
    # --------------------------------------------------
    print('Multi-class classifier - one versus one')
    
    # Data preparation 
    pairs = []
    print(testY[0][:10])
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append([i,j])
    print(pairs[0])
    print(len(pairs))
    OVO_weight_list = []
    #len(pairs)
    for x in range(len(pairs)):
        v1,v2 = pairs[x][0],pairs[x][1] # replace with method that computes each possible combination 
        trainX_OVO, trainY_OVO = img_label_extractor(X_N_train_ones, trainY,v1,v2, trainX)
        trainY_OVO_bin = sign(trainY_OVO,v1) # v1 will always be +1 
        OVO_weight = normal_eqn_solver(trainX_OVO,trainY_OVO_bin)
        OVO_weight_list.append(OVO_weight)
        print(x)
    OVO_weight_matrixhstack = np.hstack(OVO_weight_list)
    OVO_weight_matrix = np.array(OVO_weight_list, dtype='float32').T
    
    labels, confidence = binary_classifier(OVO_weight_matrix,X_N_test_ones)
  
    pred_vec = mc_one_vs_one_classifier(labels,confidence, X_N_test_ones,pairs)

    print(f'testY  Val: {testY[0][:10]}')
    print(f'Prediction: {np.reshape(pred_vec, (1,10000))[0][:10]}')
    x = np.reshape(pred_vec, (1,10000))
    confusion_matrix(np.transpose(testY), pred_vec)
    

    # --------------------------------------------------

    #results_bin = np.where(results >= 0, 1, -1).T

    
    #trainX_norm_nonzero_rqst, trainY_rqst = img_label_extractor(trainX_norm_nonzero, trainY, classifier)
    # it should be with the entire trainY, not just the requested labeled trainY 

    #weights = normal_eqn_solver(trainX_norm_nonzero_rqst, trainY)
    
    
    # Now with weighted matrix 
    # results = np.dot(norm_nonzero_img, weighted_matrix)


if __name__ == "__main__":
    main()
    
