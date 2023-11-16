import scipy.io
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# used to flatten array into 1D list -> img_label_extractor
from itertools import chain 
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
    return norm_nonzero_img
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
    info(idx,0)
    #visualize(rqst_img,5)
    return rqst_img, idx

    # Check on edge case failure of img_data request pulling same data 
    '''
    
    if np.array_equal(rqst_img, rqst_img5):
        print('true')
    else:
        print("fale")
    #print(rqst_img[5])
    '''  
def normal_eqn_solver(rqstd_img: np.array, idx: np.array) -> np.array:
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
    
    # IMPORTANT - idx may not be correct, idx as a numpy 5958x1 array or as a list of same values... check this with someon... 
    weighted_vector = np.dot(np.linalg.pinv(rqstd_img), idx)
    info(weighted_vector,0)





def main():
    trainX_normal = mat['trainX']/255
    norm_nonzero_img = zero_remover_norm(mat['trainX'])
    rqstd_img, idx = img_label_extractor(norm_nonzero_img, mat['trainY'], 2)
    weighted_matrix = normal_eqn_solver(rqstd_img, idx)


    #normal_eqn_solver(norm_nonzero_img, mat['trainY'])
if __name__ == "__main__":
    main()
    
