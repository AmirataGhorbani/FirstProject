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
    epsilon=2
    gooz_max=20
    k_max=101
    num_correl_iter=1000
    num_ordered_line_segments=100
    w=32
    h=32
    c=3
    num_test_adversaries=200
    correl1 = np.zeros((gooz_max,num_correl_iter))
    correl2 = np.zeros((gooz_max,num_correl_iter))
    correl3 = np.zeros((gooz_max,num_correl_iter))
    list_adversary1=[]
    list_adversary2=[]
    list_adversary3=[]
    dic={}
    #with open("dics/synthetic_data_intertwindness_adversary.pkl",'rb') as input:
        #dic=pickle.load(input)
    #correl1,list_adversary1=dic["1layer"]
    #correl2,list_adversary2=dic["1layer"]
    #correl3,list_adversary3=dic["1layer"]
    for gooz in range(gooz_max):
        gooz_dic={"round":gooz}
        with open("dics/gooz_synthetic{}.pkl".format(epsilon), 'wb') as output:
            pickle.dump(gooz_dic,output,pickle.HIGHEST_PROTOCOL)
        print(gooz)
        num_centers=40 #number of datapoint clusters
        intertwindness=0 #probability of a datapoint occuring in a center
        #with different label
        num_data=500 #number of datapoints in each cluster
        sigma=10+10*gooz #the standard deviation of clusters
        num_classes=10 
        centers = 256*np.random.uniform(size=(num_centers,w,h,c))
        labels=np.arange(num_classes)
        X = np.zeros((num_data*num_centers,w,h,c))
        Y = np.zeros(num_data*num_centers)
        for i in range(num_centers):
            X[i*num_data:(i+1)*num_data]=centers[i]+\
            sigma*np.random.normal(size=(num_data,w,h,c))
            toss = np.greater(np.random.uniform(size=num_data),
                              intertwindness)
            label=i%num_classes
            Y[i*num_data:(i+1)*num_data]=toss*label+\
            np.random.choice(labels,num_data)*(1-toss)
        indexes=np.arange(num_centers*num_data)
        np.random.shuffle(indexes)
        mean_image=np.mean(X,0)
        X=np.clip(X,0,256)
        X-=mean_image
        X=X[indexes]
        y=Y[indexes]
        train_size=int(0.9*num_data*num_centers)
        val_size=num_data*num_centers-train_size
        X_train=X[:train_size]
        y_train=y[:train_size]
        X_val=X[-val_size:]
        y_val=y[-val_size:]
        dictionary=analyze_distance(X_val,X_train)
        args_data=dictionary["args"]
        dist_data=dictionary["sorted"]
        temp_holder=np.zeros((val_size,k_max))
        for i in range(val_size):
            for k in range(1,k_max+1):
                temp_holder[i,k-1]=np.sum(y_train[args_data[i,:k]]==y_val[i])\
                *1./k
        dic["{}".format(sigma)]=np.mean(temp_holder,axis=0)

    #____________1 layer network_____________________________________
        tf.reset_default_graph()
        network = CNN_SC(network_name="network3", num_conv_layers=1,
                         num_forward_layers=1, input_shape=[w,h,c],
                         reg = 1e-6, num_classes=num_classes, 
                         kernel_sizes=[5], hidden_sizes=[1024],
                         pool_sizes=[8], padding = "same", 
                         path ="saved_networks/synthetic1",dims=[64], 
                         learning_rate = 1e-4, batch_norm = True, 
                         dropout = 1)
        network.optimize((X_train,y_train), (X_val,y_val),
                         epochs=min(5+10*gooz,50), load = False, save = True, 
                         verbose=False)
        NET=network
        orig_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        score_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        sess=get_session()
        saver=tf.train.Saver()
        saver.restore(sess,NET.path)
        confidence_adversaries=[]
        chagned_confidence_adversaries=[]
        robustness=[]
        number_correct_predicted=0
        for i in range(num_test_adversaries):
            output=network.adversary(X_val[i],mean_image,epsilon,
                                     y_val[i],max_iter=30,
                                     sess=sess,verbose=False,alpha=1,
                                     mode="sign")
            if not output==None:
                _,new_label,confidence, changed_confidence=output
                chagned_confidence_adversaries.append(changed_confidence)
                number_correct_predicted+=1
                if new_label:
                    confidence_adversaries.append(confidence)
            
            robustness.append(network.robustness(X_val[i],mean_image,epsilon,
                                                 y_val[i],sess=sess))
        list_adversary1.append((number_correct_predicted,
                               confidence_adversaries,
                               chagned_confidence_adversaries,
                               robustness))
        for i in range(num_correl_iter):
            args1 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            args2 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            points1 = X_val[args1]
            points2 = X_val[args2]
            orig_distance[i] = np.linalg.norm\
            (np.reshape(points1,[-1,w*c*h])-\
             np.reshape(points2,[-1,w*c*h]), axis=1)
            points_mapped1 = NET.scores(points1, sess)
            points_mapped2 = NET.scores(points2, sess)
            score_distance[i] = np.linalg.norm\
            (points_mapped1-points_mapped2,axis=1)
        for i in range(num_correl_iter):
            correl1[gooz,i] = stats.pearsonr(orig_distance[i],
                                             score_distance[i])[0]



    #____________2 layer network_____________________________________
        tf.reset_default_graph()
        network = CNN_SC(network_name="network3", num_conv_layers=2,
                          num_forward_layers=1,input_shape=[w,h,c], 
                          reg = 1e-4,num_classes=num_classes, kernel_sizes=[5,5],
                          hidden_sizes=[1024], pool_sizes=[2,2], 
                          padding = "same", path ="saved_networks/synthetic2",
                          dims=[64,128], learning_rate = 5e-4, 
                          batch_norm = True, dropout = 1)
        network.optimize((X_train,y_train), (X_val,y_val), epochs=min(5+10*gooz,50),
                          load = False, save = True, verbose=False)
        NET=network
        orig_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        score_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        sess=get_session()
        saver=tf.train.Saver()
        saver.restore(sess,NET.path)
        arg_adversaries=[]
        confidence_adversaries=[]
        chagned_confidence_adversaries=[]
        robustness=[]
        number_correct_predicted=0
        for i in range(num_test_adversaries):
            output=network.adversary(X_val[i],mean_image,epsilon,
                                     y_val[i],max_iter=30,
                                     sess=sess,verbose=False,alpha=1,
                                     mode="sign")

            if not output==None:
                _,new_label,confidence, changed_confidence=output
                chagned_confidence_adversaries.append(changed_confidence)
                number_correct_predicted+=1
                if new_label:
                    confidence_adversaries.append(confidence)
            robustness.append(network.robustness(X_val[i],mean_image,epsilon,
                                                 y_val[i],sess=sess))
        list_adversary2.append((number_correct_predicted,
                               confidence_adversaries,
                               chagned_confidence_adversaries,
                               robustness))
        for i in range(num_correl_iter):
            args1 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            args2 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            points1 = X_val[args1]
            points2 = X_val[args2]
            orig_distance[i] = np.linalg.norm\
            (np.reshape(points1,[-1,w*h*c])-\
             np.reshape(points2,[-1,w*h*c]), axis=1)
            points_mapped1 = NET.scores(points1, sess)
            points_mapped2 = NET.scores(points2, sess)
            score_distance[i] = np.linalg.norm\
            (points_mapped1-points_mapped2,axis=1)
        for i in range(num_correl_iter):
            correl2[gooz,i] = stats.pearsonr(orig_distance[i],
                                             score_distance[i])[0]





    #____________3 layer network_____________________________________
        tf.reset_default_graph()
        network = CNN_SC(network_name="network3", num_conv_layers=3,
                         num_forward_layers=1,input_shape=[w,h,c],
                         reg = 1e-2,num_classes=num_classes,
                         kernel_sizes=[5,5,5],
                         hidden_sizes=[1024], pool_sizes=[2,2,2],
                         padding = "same",
                         path ="saved_networks/synthetic3",
                         dims=[64,128,256], learning_rate = 1e-3,
                         batch_norm = True, dropout = 1)
        network.optimize((X_train,y_train), (X_val,y_val),
                         epochs=min(5+10*gooz,50),
                         load = False, save = True, verbose=False)
        NET=network
        orig_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        score_distance=np.zeros((num_correl_iter,num_ordered_line_segments))
        sess=get_session()
        saver=tf.train.Saver()
        saver.restore(sess,NET.path)
        arg_adversaries=[]
        confidence_adversaries=[]
        chagned_confidence_adversaries=[]
        robustness=[]
        number_correct_predicted=0
        for i in range(num_test_adversaries):
            output=network.adversary(X_val[i],mean_image,epsilon,
                                     y_val[i],max_iter=30,
                                     sess=sess,verbose=False,alpha=1,
                                     mode="sign")

            if not output==None:
                _, new_label, confidence, changed_confidence = output
                chagned_confidence_adversaries.append(changed_confidence)
                number_correct_predicted+=1
                if new_label:
                    confidence_adversaries.append(confidence)
            robustness.append(network.robustness(X_val[i],mean_image,epsilon,
                                                 y_val[i],sess=sess))
        list_adversary3.append((number_correct_predicted,
                               confidence_adversaries,
                               chagned_confidence_adversaries,
                               robustness))
        for i in range(num_correl_iter):
            args1 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            args2 = np.random.choice(val_size,num_ordered_line_segments,replace=False)
            points1 = X_val[args1]
            points2 = X_val[args2]
            orig_distance[i] = np.linalg.norm\
            (np.reshape(points1,[-1,w*h*c])-\
             np.reshape(points2,[-1,w*h*c]), axis=1)
            points_mapped1 = NET.scores(points1, sess)
            points_mapped2 = NET.scores(points2, sess)
            score_distance[i] = np.linalg.norm\
            (points_mapped1-points_mapped2,axis=1)
        for i in range(num_correl_iter):
            correl3[gooz,i] = stats.pearsonr(orig_distance[i],
                                             score_distance[i])[0]
        dic["1layer"]=(correl1,list_adversary1)
        dic["2layer"]=(correl2,list_adversary2)
        dic["3layer"]=(correl3,list_adversary3)
        with open("dics/synthetic_data_intertwindness_adversary.pkl", 'wb') as output:
                pickle.dump(dic,output,pickle.HIGHEST_PROTOCOL)            

           

