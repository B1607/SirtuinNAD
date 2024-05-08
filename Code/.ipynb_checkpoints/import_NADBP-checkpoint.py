import numpy as np
import tensorflow as tf
import gc


def MCNN_data_load(NUM_CLASSES,NUMDEPENDENT,dataset):
    MAXSEQ=NUMDEPENDENT*2+1
    path_x_train = "../Dataset/Train/data.npy"
    path_y_train = "../Dataset/Train/label.npy"
    print(path_x_train)
    print(path_y_train)
    x,y=data_load(path_x_train,path_y_train,NUM_CLASSES)
    path_x_test =  "../Dataset/"+dataset+"/data.npy"
    path_y_test =  "../Dataset/"+dataset+"/label.npy"
    x_test,y_test=data_load(path_x_test,path_y_test,NUM_CLASSES)
    print(path_x_test)
    print(path_y_test)
    return(x,y,x_test,y_test)

def data_load(x_folder, y_folder,NUM_CLASSES,):
    x_train=np.load(x_folder)
    y_train=np.load(y_folder)
    y_train = tf.keras.utils.to_categorical(y_train,NUM_CLASSES)
    gc.collect()
    return x_train, y_train