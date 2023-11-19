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
def prediction_calc(arr: np.array) -> np.array:
    '''
    Calculates predictions for OneVsOne Classifier
    by tabulating the votes in each row. Ties 
    are broken by picken the index of first apperance of 
    equal tally.
    
    
    Parameters
    ---------
    arr: np.array
        Holds votes of each, expected 10K x 45 
        
        
        
    Returns 
    -------
    predictions: np.array
        Holds result of tabulated vote, ie classifier image prediction label

    ''' 
    row,columns = np.shape(arr)

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
    '''
    Display Data - Debugger had high load time in some instances
    
    
    Parameters
    ---------
    values: Any
        Typically np.array, chosen item to display data of
    printr: bool
        to print or not print, that is the question
    
    Returns 
    -------

    ''' 
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
    trainX_norm_nonzero = img_array_zero/255
    

    return trainX_norm_nonzero, idx
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
def img_label_extractor(norm_nonzero_img: np.array, train_label: np.array, label_rqst1: int, label_rqst2:int) -> np.array:
    '''
    Purpose: Extracts the img_data corresponding to the desired label 
    
    Parameters
    ----------
    norm_nonzero_img: np.array 
        Normalized and zeroed columns removed dataset 
    label_array: np.array
        True labels of our test image dataset 
    label_rqst1/2: ints
        The requested image labels to extract from corresponding 
        norm_nonzero_img array
        
    Returns 
    ----------
    labeled_img_data: np.array
        Extracted img data that is only of corresponding requested label  
        

    ''' 

    
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
    #-------------------------------------
 
    idx1_r = np.argwhere(np.all(train_label[..., :] != label_rqst1,axis = 0 ))
    idx2_r = np.argwhere(np.all(train_label[..., :] != label_rqst2,axis = 0 ))
    idx1_rflat = list(chain.from_iterable(idx1_r.tolist()))
    idx2_rflat = list(chain.from_iterable(idx2_r.tolist()))


    idx_r_common = [common for common in idx1_rflat if common in idx2_rflat]

    trainY_OVO = np.delete(train_label,idx_r_common)
    #-------------------------------------

    return rqst_img_both, trainY_OVO

def normal_eqn_solver(trainX_norm_nonzero_ones: np.array, trainY: np.array, bin_class:int) -> np.array:
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


    trainY_bin = sign(trainY, bin_class)

    #weights = np.linalg.pinv(trainX_norm_nonzero_ones) @ trainY_bin # see if necessary or can do np.dot(X,Y), resource consumption
    weights = np.dot(np.linalg.pinv(trainX_norm_nonzero_ones),trainY_bin) # see if necessary or can do np.dot(X,Y), resource consumption

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
    #confidence = X_N_test_ones @ weights # see if necessary or can do np.dot(X,Y), resource consumption
    confidence = np.dot(X_N_test_ones, weights) # see if necessary or can do np.dot(X,Y), resource consumption

    classifier_labels = np.sign(confidence)
    return classifier_labels, confidence

def one_v_all_weight_calcs(trainY: np.array, X_N_train_ones: np.array)->np.array:
    '''
    Calculates the weights for the one_v_all classifier
    
    
    Parameters
    ---------
    trainY: np.array
        Labels of training dataset
    X_N_train_ones: np.array
        Image data that is normalized, with ones column added
        onto the end, and 0 columns removed
        
    Returns 
    -------
    weight_matrix: np.array
        Solution of the normal equation, weights of each pixel 
        information for ova classifier
    ''' 
    
    
    weight_list = []
    for x in range(10):
        weights = normal_eqn_solver(X_N_train_ones, trainY,x)
        weight_list.append(weights)
        print(x) # Indicates how many loops till completion 
    weight_matrix = np.hstack(weight_list)
    return weight_matrix
def mc_one_vs_all_classifier(trainY: np.array, weights: np.array, X_N_test_ones: np.array):
    '''
    Multiclass classifier model built upon one_vs_all weights 
    
    
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
    
    #confidence= X_N_test_ones @ weights # see if necessary or can do np.dot(X,Y), resource consumption
    confidence = np.dot(X_N_test_ones,weights) # see if necessary or can do np.dot(X,Y), resource consumption


    # Finding the indices of the maximum values in each row
    max_indices = np.argmax(confidence, axis=1)
    mc_ova_labels = []
    mc_ova_labels.append(np.argmax(confidence,axis=1))

    return mc_ova_labels
def pair_calc():
    '''
    Calculates unique pairs from numbers 0-9
    
    
    Parameters
    ---------
    N/A
        
        
    Returns 
    -------
    45 unique pairs of numbers 0-9
    ''' 
    pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            pairs.append([i,j])
    print(pairs[0])
    print(len(pairs))
    return pairs
def one_vs_one_weight_calc(X_N_train_ones: np.array,trainY:np.array) -> np.array:
    '''
    Calculates the weights for the one vs one classifier
    
    
    Parameters
    ---------
    trainY: np.array
        Labels of training dataset
    X_N_train_ones: np.array
        Image data that is normalized, with ones column added
        onto the end, and 0 columns removed
    t
    Returns 
    -------
    OVO_weight_matrix: np.array
        Solution of the normal equation, weights of each pixel 
        information for OVO classifier 
    ''' 
    
    OVO_weight_list = []
    pairs = pair_calc()
    for x in range(len(pairs)):
        v1,v2 = pairs[x][0],pairs[x][1] # replace with method that computes each possible combination 
        trainX_OVO, trainY_OVO = img_label_extractor(X_N_train_ones, trainY,v1,v2)
        OVO_weight = normal_eqn_solver(trainX_OVO,trainY_OVO, v1)
        OVO_weight_list.append(OVO_weight)
        print(x)
    OVO_weight_matrix = np.array(OVO_weight_list, dtype='float32').T

    return OVO_weight_matrix, pairs
def mc_one_vs_one_classifier(labels: np.array,confidence,X_N_test_ones, pairs):
    '''
    Multiclass classifier model built upon one_vs_one weights 
    
    
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

def error_calc(testY, classifier_labels) -> int:
    
############################### DEPRICATED FUNCTION ###############################
                             # Kept for posterity sake 
###################################################################################
    '''
    Error rate calculation     
    
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
        

    
   
    error = 0 

    # Error Calculation 
    # -------------------------------------------
    r,c = np.shape(classifier_labels)
    # Ensure c is ranging over the correct value 
    for x in range(r):
        if testY[x] != classifier_labels[x]:
            error = error+1 
    err_perc = error/r
    #print("Error Percent: " + str(err_perc) + " Error Norm: " + str(error))
    
    
    # -------------------------------------------
    
    
   
    
    return err_perc

def bin_confusion(yTrain_bin, bin_predict):
    
    # True Positive, True Negative, False Positive, and False Negative var initializations
                 
    acc_data = [0,0,0,0]   
    
    tp, fp, tn, fn = 0,0,0,0
    # test_pairs = [(1,1), (-1,-1), (1,-1), (-1, 1)]
    #info(yTrain_bin[:50][0],1)
    #info(bin_predict[:10][0],1)
   
    # r,c = np.shape(bin_predict)
    r,c = np.shape(yTrain_bin)
    #info(yTrain_bin,0)
    predic_true = np.count_nonzero(bin_predict==1)
    predic_false = np.count_nonzero(bin_predict==-1)
    for idx, true_val in enumerate(yTrain_bin):
        if true_val == 1:
            if bin_predict[idx] == 1:
                tp =tp + 1
            else: 
                fn = fn + 1
        else: # true_val == -1 
            if bin_predict[idx] == -1:
                tn = tn+1 
            else: # bin_predict[idx] = 1
                fp = fp + 1 
    
    error = (fp+fn)/r
    precision = tp/(tp+fp)
    print(f'Error: {error} \nPrecision: {precision}')
    print(f'True Positive Rate: {tp/predic_true} \nTrue Negative: {tn/predic_false}\nFalse Positive: {fp/predic_false}')
    arr1 = yTrain_bin.reshape(c,r)
    arr2 = bin_predict.reshape(c,r)

    y_actu = pd.Series(arr1.flatten(), name='Actual')
    y_pred = pd.Series(arr2.flatten(), name='Predicted')
    df_conf = pd.crosstab(y_actu, y_pred, margins=True)
    print(df_conf)
    #df_perc = pd.crosstab(y_actu, y_pred, normalize = 'index')
    
    
    
    
    
    
    #print(df_confusion)
    return df_conf
    '''
    # Old method but stored in case of use for future confusion matrix calculations 
        # tp = np.logical_and(yTrain_bin, bin_predict)
    for idx, pair in enumerate(test_pairs):
        temp = (yTrain_bin == pair[0]) & (bin_predict == pair[1])
        acc_data[idx] = np.sum(temp)
    print(acc_data)
    
    # Worked but crashed too often 
    

    
        for idx in range(r):
        if yTrain_bin[idx] == 1:
            if yTrain_bin[idx] == 1:
                acc_data[0] = acc_data[0] + 1
            else:
                acc_data[3] = acc_data[3] + 1
        else:
            if yTrain_bin[idx] == 1:
                acc_data[3] = acc_data[3] + 1
            else:
                acc_data[2] = acc_data[2] + 1


    '''
def main():

############################################################################################################
########################### Data setup #####################################################################
############################################################################################################    
   # --------------------------------------------------
    trainX = mat['trainX']
    trainY = mat['trainY']

    X_N_train, rmved_c = zero_remover_norm(trainX)
    
    r,c = np.shape(X_N_train)
    train_ones_column = np.ones((r,1))
    X_N_train_ones = np.hstack((X_N_train,train_ones_column))

    
    
    # --------------------------------------------------

# testX manipulation - normalizes, removes same 0 columns as trainX, and adds 1 column - should recognifugre to be 
    # --------------------------------------------------
    testX_normal = mat['testX']/255
    testY = mat['testY']
    X_N_test = np.delete(testX_normal, rmved_c, axis=1)
    r = 10000
    test_ones_column = np.ones((r,1))
    X_N_test_ones = np.hstack((X_N_test,test_ones_column))
    # X_N = Normalized, Zero Columns removed, and ones column tacked on the end 
    # --------------------------------------------------
    
    
############################################################################################################
########################### Randomized Feature Based Least Square Classifier Set Up ########################
############################################################################################################
   # --------------------------------------------------
    W = np.random.normal(0,1,(717,1000)) # no .T like in lab doc 
    b = np.random.normal(0,1,(1,1000))
    
    # Apply ones column last as to not alter 1's column
    # Use normalized & 0's removed first 
    X_train_rand = (X_N_train @ W) + b
    
    X_train_rand = 1/(1+np.exp(-1*X_train_rand))
    #nlm_train = [X_train_rand, 1/(1+np.exp(-1*X_train_rand)),np.sin(X_train_rand), np.maximum(X_train_rand,0,X_train_rand)]
    
    
    
    X_test_rand = (X_N_test @ W) + b
    #nlm_test = [X_test_rand, 1/(1+np.exp(-1*X_test_rand)),np.sin(X_test_rand), np.maximum(X_test_rand,0,X_test_rand)]
    X_test_rand =  1/(1+np.exp(-1*X_test_rand))
    # Add ones column to each index of nlm_train & nlm_test
    #for idx in range(len(nlm_train)):
    #    nlm_train[idx] = np.hstack((nlm_train[idx]),train_ones_column)
    #    nlm_test[idx] = np.hstack((nlm_test[idx]),test_ones_column)


    # Comment or uncomment following line to use feature mapping or not 
    #X_N_train_ones = np.hstack(((X_train_rand),train_ones_column))
    #X_N_test_ones = np.hstack(((X_test_rand),test_ones_column))

    # --------------------------------------------------
 
############################################################################################################
########################### Binary Classifier - N Versus All ###############################################
############################################################################################################
    # -------------------------------------------------
    

    print('Once counter hits 10 program will have run through all 10 binary classifier \nConfusion Matrices & Errors for each classifier will output  each iteration')
    for bin_class in range(10):
    
        # Calculates weights 
        bin_weight = normal_eqn_solver(X_N_train_ones,trainY, bin_class)
        
        testY_bin = sign(testY,bin_class)
        
        # Classifies index as bin_class or not bin_class 
        bin_classifier_predictions, confidence = binary_classifier(bin_weight,X_N_test_ones)   

        # Error, Precision, & Confusion Matrix Calculations
        print(f'Binary Classifier: {str(bin_class)}') 
        bin_confusion(testY_bin,bin_classifier_predictions)
        
    
  
    
    # --------------------------------------------------


# Multi-class one-versus all classifier 
    # --------------------------------------------------
    
    
    # Calculates weights
    weight_matrix = one_v_all_weight_calcs(trainY, X_N_train_ones)
    
    # Creates predictions 
    mc_classifier_predictions = mc_one_vs_all_classifier(trainY, weight_matrix, X_N_test_ones)
    info(mc_classifier_predictions,0)
    # Calculates Error 
    one_v_all_error = error_calc(np.transpose(testY), np.transpose(mc_classifier_predictions))
    
    print(one_v_all_error)
    
    # --------------------------------------------------

  
# Multi-class one versus one classifier 

    # --------------------------------------------------
    print('Multi-class classifier - one versus one')
     
    
    
    # Calculate weights & labels 
    OVO_weight_matrix, pairs = one_vs_one_weight_calc(X_N_train_ones,trainY)
    labels, confidence = binary_classifier(OVO_weight_matrix,X_N_test_ones)
  
  
    # Produce predictions 
    pred_vec = mc_one_vs_one_classifier(labels,confidence, X_N_test_ones,pairs)

    print(f'testY  Val: {testY[0][:10]}')
    print(f'Prediction: {np.reshape(pred_vec, (1,10000))[0][:10]}')
    x = np.reshape(pred_vec, (1,10000))
    
    # Calculate Error 
    onevoneerror = error_calc(np.transpose(testY), pred_vec)


    print(f'One V one Error: {onevoneerror}')
    print(f'One V all Error: {one_v_all_error}')
    #print(f'Bin Error: {bin_error}')

    
    # --------------------------------------------------



if __name__ == "__main__":
    main()
    