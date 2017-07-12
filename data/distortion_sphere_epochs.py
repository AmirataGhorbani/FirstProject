import matplotlib
matplotlib.use('Agg') 
import numpy as np
import tensorflow as tf
import random
import pickle
import scipy
import matplotlib.pyplot as plt
from d_u import load_CIFAR10
import structures
from amirata_functions import *
from pylab import rcParams
from tqdm import tqdm_notebook as tqdm
from CNN_class import CNN_SC
import scipy
import scipy.stats as stats
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
if __name__ == "__main__":
    def save_object(object, filename):
        with open(filename, 'wb') as output:
            pickle.dump(object,output,pickle.HIGHEST_PROTOCOL)

    def load_object(object, fielname):
        with open(fielname, 'rb') as input:
            object = pickle.load(input)
            
    data,aux_data = get_CIFAR10_data("../datasets/cifar-10-batches-py")
    mean_image=data["mean_image"]
    X_train=data["X_train"]
    y_train=data["y_train"]
    y_val = data["y_val"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    X_train1 = X_train[:25000]
    y_train1 = y_train[:25000]
    X_val1 = X_val[:2000]
    y_val1 = y_val[:2000]
    X_train2 = X_train[25000:]
    y_train2 = y_train[25000:]
    X_val2 = X_val[5000:]
    y_val2 = y_val[5000:]
    X_train.shape
    training_data = (X_train,y_train)
    training_data1 = (X_train1, y_train1)
    training_data2 = (X_train2, y_train2)
    validation_data = (X_val,y_val)
    validation_data1 = (X_val1,y_val1)
    validation_data2 = (X_val2,y_val2)
    test_data = (X_test, y_test)
    
    tf.reset_default_graph()
    clustering = False
    n=1000
    N=1000
    iters=20
    conds=np.zeros((iters,N))
    dic={}
    best_val = 0.
    network = CNN_SC(network_name="network1", num_conv_layers=1, num_forward_layers=1,
                     input_shape=[32,32,3], reg = 1e-6,num_classes=10, kernel_sizes=[5], hidden_sizes=[1024],
                     pool_sizes=[8], padding = "same", path ='saved_networks/goozu1',
                     dims=[64], learning_rate = 1e-4, batch_norm = True, dropout = 1,initialize=True)
    NET=network
    sess=get_session()
    saver = tf.train.Saver()
    saver.restore(sess,NET.path)  
    val = X_val[2000:]
    conds[0]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,
                               hidden_size=1024,verbose=True,X_val=val,sess=sess)
    for cntr in range(1,iters):
        network.optimize(training_data, validation_data1, epochs=1, load = True, save = True,
                         save_always=True)

        saver.restore(sess,NET.path)  
        val = X_val[2000:]
        conds[cntr]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,hidden_size=1024,
                                      verbose=True,X_val=val,sess=sess)
    
        dic["conds"]=conds
        save_object(dic,"1layer_training_sphere.pkl")
    tf.reset_default_graph()
    clustering = False
    conds=np.zeros((iters,N))
    dic={}
    best_val = 0.
    network = CNN_SC(network_name="network1", num_conv_layers=2, num_forward_layers=1,input_shape=[32,32,3],
                     reg = 1e-4,num_classes=10, kernel_sizes=[5,5], hidden_sizes=[1024],
                     pool_sizes=[2,2], padding = "same", path ='saved_networks/goozu2',
                      dims=[64,128], learning_rate = 5e-4, batch_norm = True, dropout = 1,initialize=True)
    NET=network
    sess=get_session()
    saver = tf.train.Saver()
    saver.restore(sess,NET.path)  
    val = X_val[2000:]
    conds[0]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,hidden_size=1024,verbose=False,X_val=val,sess=sess)
    for cntr in range(1,iters):
        network.optimize(training_data, validation_data1, epochs=1, load = True, save = True,
                         save_always=True)

        saver.restore(sess,NET.path)  
        val = X_val[2000:]
        conds[cntr]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,hidden_size=1024,
                                      verbose=False,X_val=val,sess=sess)
        dic["conds"]=conds
        save_object(dic,"2layer_training_sphere.pkl")
    tf.reset_default_graph()
    clustering = False
    conds=np.zeros((iters,N))
    dic={}
    best_val = 0.
    network = CNN_SC(network_name="network1", num_conv_layers=3, num_forward_layers=1,input_shape=[32,32,3],
                     reg = 1e-2,num_classes=10, kernel_sizes=[5,5,5], hidden_sizes=[1024],
                     pool_sizes=[2,2,2], padding = "same", path ='saved_networks/goozu3',
                      dims=[64,128,256], learning_rate = 1e-3, batch_norm = True, dropout = 1,initialize=True)
    NET=network
    sess=get_session()
    saver = tf.train.Saver()
    saver.restore(sess,NET.path)  
    val = X_val[2000:]
    conds[0]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,hidden_size=1024,
                               verbose=False,X_val=val,sess=sess)
    for cntr in range(1,iters):
        network.optimize(training_data, validation_data1, epochs=1, load = True, save = True,
                         save_always=True)

        saver.restore(sess,NET.path)  
        val = X_val[2000:]
        conds[cntr]=sphere_distortion(1,N,1000,NET,w=32,h=32,c=3,hidden_size=1024,
                                      verbose=False,X_val=val,sess=sess)
    
        dic["conds"]=conds
        save_object(dic,"3layer_training_sphere.pkl")
