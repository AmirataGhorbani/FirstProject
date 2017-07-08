import numpy as np
import tensorflow as tf
import random
import pickle
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
    data,aux_data = get_CIFAR10_data("../datasets/cifar-10-batches-py")
    mean_image=data["mean_image"]
    X_val=data["X_val"][9000:]
    y_val=data["y_val"][9000:]
    dictionary=analyze_distance(X_val,data["X_train"])
    dists=dictionary["dist"]
    args=dictionary["args"]
    tf.reset_default_graph()
    training_data = (data["X_train"],data["y_train"])
    validation_data1 = (aux_data["X_val1"][1:100],aux_data["y_val1"][1:100])
    network3 = CNN_SC(network_name="network3", num_conv_layers=3,
                      num_forward_layers=1,input_shape=[32,32,3], 
                      reg = 1e-2,num_classes=10, kernel_sizes=[5,5,5],
                      hidden_sizes=[1024], pool_sizes=[2,2,2], 
                      padding = "same", path ="saved_networks/1C_3L_F",
                      dims=[64,128,256], learning_rate = 1e-3, 
                      batch_norm = True, dropout = 1)
    network3.optimize(training_data, validation_data1, epochs=0,
                      load = True, save = True, verbose=True)
    
    number_correct_predicted=0
    number_adversary_found=0
    number_assumption_true=0
    sess=get_session()
    f=np.zeros(10)
    tf.train.Saver().restore(sess,network3.path)
    epsilon=2
    for i in range(1000):
        print(i)
        output=network3.adversary(X_val[i],mean_image,epsilon,
                                         y_val[i],max_iter=30,
                                         sess=sess,verbose=False,alpha=1,
                                         mode="sign")
        if not output==None:
            adv,changed_label,confidence=output
            number_correct_predicted+=1
            if changed_label:
                number_adversary_found+=1
                true_label=y_val[i]
                new_label=np.argmax(network3.feedforward(adv,sess),1)
                places=np.where(data["y_train"][args[i]]==new_label)[0]
                for number,place_arg in enumerate(places[:10]):
                    found=True
                    for j in np.linspace(0,1,100):
                        point=(1-j)*adv+(j)*data["X_train"][args[i,place_arg]]
                        point_label=np.argmax(network3.feedforward(point,sess),1)
    #                     print(i,place_arg,j,point_label)
                        if point_label!=new_label:
                            found=False
                            break
                    if found:
                        number_assumption_true+=1
                        f[number]+=1
                        break
        dic={"number_assumption_true":number_assumption_true,"f":f,
             "number_adversary_found":number_adversary_found,
             "number_correct_predicted":number_correct_predicted}
        with open("dics/around_adversaries.pkl", 'wb') as output:
                    pickle.dump(dic,output,pickle.HIGHEST_PROTOCOL)            