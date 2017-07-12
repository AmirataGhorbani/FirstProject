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
    tf.reset_default_graph()
    network3 = CNN_SC(network_name="network3", num_conv_layers=3,
                  num_forward_layers=1,input_shape=[32,32,3], 
                  reg = 1e-2,num_classes=10, kernel_sizes=[5,5,5],
                  hidden_sizes=[1024], pool_sizes=[2,2,2], 
                  padding = "same", path ="saved_networks/1C_3L_F",
                  dims=[64,128,256], learning_rate = 1e-3, 
                  batch_norm = True, dropout = 1)
    network3.optimize((data["X_train"],data["y_train"]),
                      (aux_data["X_val1"][1:10],aux_data["y_val1"][1:10]), 
                      epochs=0,load = True, save = True, verbose=False)
    sess=get_session()
    saver=tf.train.Saver()
    saver.restore(sess,network3.path)
    for epsilon in [8]:
#     dic1={}
        dic2={}
        for i in range(1000):
            gooz_dic={"round":i}
            with open("gooz{}.pkl".format(epsilon), 'wb') as output:
                pickle.dump(gooz_dic,output,pickle.HIGHEST_PROTOCOL)
    #         output1=network3.adversary(data["X_val"][9000+i],mean_image,epsilon,
    #                                   data["y_val"][9000+i],max_iter=30,
    #                                   sess=sess,verbose=False,alpha=1)
            output2=network3.adversary(data["X_val"][9000+i],mean_image,epsilon,
                                      data["y_val"][9000+i],max_iter=30,
                                      sess=sess,verbose=False,alpha=1,mode="sign")

    #         if output1!=None:
    #             dic1["{}".format(i)]={"adversary":output1[0],
    #                                 "new_label":output1[1],
    #                                 "confidence":output1[2]}
            if output2!=None:    
                dic2["{}".format(i)]={"adversary":output2[0],
                                    "new_label":output2[1],
                                    "confidence":output2[2]}

    #     with open("dics/CIFAR10_adversaries_BIM_epsilon_{}.pkl".format(epsilon), 'wb') as output:
    #         pickle.dump(dic1,output,pickle.HIGHEST_PROTOCOL)
        with open("dics/CIFAR10_adversaries_sign_epsilon_{}.pkl".format(epsilon), 'wb') as output:
            pickle.dump(dic2,output,pickle.HIGHEST_PROTOCOL)
