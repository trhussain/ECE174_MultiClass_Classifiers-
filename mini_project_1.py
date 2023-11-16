import scipy.io
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# used to flatten array into 1D list -> img_label_extractor
from itertools import chain 

from numpy.linalg import inv
from numpy.linalg import matrix_rank
mat = scipy.io.loadmat('mnist.mat')

np.set_printoptions(threshold=np.inf)  # Set threshold to infinity to print all elements


def visualize(norm_trainX: np.array, index: int) -> None:
    '''
    Purpose: To visualize img data for manual verification of training, and fun 
    
    Parameters
    ----------
    index: int 
    Index of requested image data 
    norm_trainX: np.array
    Normalized pixel information 
    
    Returns 
    -------
    Outputs requested image, no return value 
    
    
    
    '''
    
    # Extracts requested row from norm_trainX, reshapes into 28x28 pixel image and displays
    # Difference between using the normalized dataset and non-normalized is nothing...? 
    number = norm_trainX[index].reshape((28, 28)) 
    plt.imshow(number)  
    plt.axis('off')  
    plt.show()

def info(values, v):
    
    # stfu the debugger takes forever to load this is way quicker #dontjudgemepls
    print(type(values))
    print(np.shape(values))
    if v == 1:
        print(values)
def zero_remover_norm(img_array: np.array) -> np.array:
    '''
    Purpose: To remove the 0 columns of inputted numpy array 
    
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
    
 
    trainY_bin = np.where(trainY == label_rqst, 1, -1).T
    
    return trainY_bin    
def img_label_extractor(norm_nonzero_img: np.array, train_label: np.array, label_rqst: int) -> np.array:
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
    
    idx = np.argwhere(np.all(train_label[..., :] == label_rqst, axis=0))
    idx_list = idx.tolist()
    idx_flat = list(chain.from_iterable(idx_list))
    rqst_img = norm_nonzero_img[idx_flat]
    #visualize(rqst_img,5)
    return rqst_img, idx_flat
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
def one_versus_all_classifier(trainY_bin: np.array, weights: np.array, norm_nonzero_img: np.array, ) -> np.array:
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
    
    classifier = np.dot(norm_nonzero_img, weights)
    max_values = np.amax(classifier, axis = 1)
    info(classifier,0)
    max_indices = np.argmax(classifier, axis=1)
    classifier_labels = max_indices.reshape(-1, 1)  # Reshape to a column vector
    return classifier_labels
    #max_xy = np.where(max_ == a.max() )
    #isualize(norm_nonzero_img, 5000)

    #results = np.square(trainY-(np.dot(np.transpose(weights),norm_nonzero_img)))

def error(trainY, classifier_labels):
    
    print(trainY)
    
def main():

    # Setup to compute the weighted matrix solution to the normal equation 
    trainX = mat['trainX']
    trainY = mat['trainY']
    testX_normal = mat['testX']/255
    testY = mat['testY']
    
    X_N_train, rmved_c = zero_remover_norm(trainX)
    
    # Need to transform into a function 
    X_N_test = np.delete(testX_normal, rmved_c, axis=1)
    r = 10000
    ones_column = np.ones((r,1))
    X_N_test_ones = np.hstack((X_N_test,ones_column))
    
    
    # X_N = Normalized, Zero Columns removed, and ones column tacked on the end 
    
    visualize(testX_normal,0)
    weight_list = []
    for x in range(10):
        trainY_bin = sign(trainY, x)
        weights = normal_eqn_solver(X_N_train, trainY_bin)
        weight_list.append(weights)
        
    weight_matrix = np.hstack(weight_list)
    info(weight_matrix,0)
    classifier_labels = one_versus_all_classifier(trainY, weight_matrix, X_N_test_ones)
    
    
    
    

    #results_bin = np.where(results >= 0, 1, -1).T

    
    #trainX_norm_nonzero_rqst, trainY_rqst = img_label_extractor(trainX_norm_nonzero, trainY, classifier)
    # it should be with the entire trainY, not just the requested labeled trainY 

    #weights = normal_eqn_solver(trainX_norm_nonzero_rqst, trainY)
    
    
    # Now with weighted matrix 
    # results = np.dot(norm_nonzero_img, weighted_matrix)


if __name__ == "__main__":
    main()
    
