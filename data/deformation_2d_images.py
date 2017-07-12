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

    w=2
    h=1
    c=1
    gooz=1
    num_centers=6
    intertwindness=0
    num_data=500
    shomar=1
    for number in range(2,3):
        for sigma in [0.03]:
            shomar+=1
            num_classes=3
            centers = np.random.uniform(size=(num_centers,w,h,c))
            centers = np.expand_dims(np.expand_dims(np.array([[0.1,0.1],[0.8,0.3],[0.3,0.8],
                                                              [0.8,0.8],[0.2,0.5],[0.5,0.3]]),-1),-1)
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

            X=np.clip(X,0,1)
            indexes=np.arange(num_centers*num_data)
            np.random.shuffle(indexes)
            X=X[indexes]
            y=Y[indexes]
            X1 = X[np.where(y==0)]
            X2 = X[np.where(y==1)] 
            X3 = X[np.where(y==2)] 
            train_size=int(0.9*num_data*num_centers)
            val_size=num_data*num_centers-train_size
            X_train=X[:train_size]
            y_train=y[:train_size]
            X_val=X[-val_size:]
            y_val=y[-val_size:]

            accuracy = 400
            number_partition=4
            points = np.zeros((accuracy**2,2,1,1))
            colors = np.zeros((accuracy**2,3))
            colors2 = np.zeros((accuracy**2,3))
            counter = -1
            for i,x in enumerate(np.linspace(0,1,accuracy)):
                for j,y in enumerate(np.linspace(0,1,accuracy)):
                    cd1=(i//25)%2
                    cd2=(j//25)%2
                    counter += 1
                    new_point=np.zeros((2,1,1))
                    new_point[0,0,0] = x
                    new_point[1,0,0] = y
                    points[counter] = new_point
                    dists1 = np.min(np.sqrt(np.sum((X1-new_point)**2,1)))
                    dists2 = np.min(np.sqrt(np.sum((X2-new_point)**2,1)))
                    dists3 = np.min(np.sqrt(np.sum((X3-new_point)**2,1)))
                    color = np.zeros(3)
                    color2 = np.zeros(3)
                    if dists2>dists1 and dists3>dists1:
                        color[0] = np.exp(-1000*dists1**2)
                    elif dists1>dists2 and dists3>dists2:
                        color[2] = np.exp(-1000*dists2**2)
                    else:
                        color[0] = np.exp(-1000*dists3**2)
                        color[2] = np.exp(-1000*dists3**2)
                    color[1] = (2*x**2+y**2)/3*(1/(1+np.exp(-2*np.min([dists1,dists2,dists3])-0.01)))
                    color2=np.array([(cd1==1)*(cd2==1),(cd1==0)*(cd2==1),(cd1==0)*(cd2==0)])
                    colors[counter] = color
                    colors2[counter] = color2
            for num_layer in [10]:
                print(num_layer)
                tf.reset_default_graph()
                net = CNN_SC(network_name="network_small", num_conv_layers=0,
                          num_forward_layers=num_layer,input_shape=[w,h,c], 
                          reg = 1e-4,num_classes=num_classes, kernel_sizes=[],
                          hidden_sizes=[10]*(num_layer-1)+[2], pool_sizes=[], 
                          padding = "same", path ="saved_networks/deform10",
                          dims=[], learning_rate = 1e-3, 
                          batch_norm = True, dropout = 1, activation="tanh")
                net.optimize((X_train,y_train), (X_val,y_val), epochs=0,
                                 load = False, save = True, verbose=False, save_always=True)
                for epoch in range(20):
                    print(epoch)
                    sess=get_session()
                    tf.train.Saver().restore(sess,net.path)
                    length = len(points)//1000
                    hidden=np.zeros((length*1000,net.hidden_sizes[-1]))
                    output=np.zeros((length*1000,net.num_classes))
                    label=np.zeros(length*1000)
                    for counter in range(length):
                        hidden[counter*1000:(counter+1)*1000] =\
                        net.scores(points[counter*1000:(counter+1)*1000],sess)
                        output[counter*1000:(counter+1)*1000] =\
                        net.feedforward(points[counter*1000:(counter+1)*1000],sess)
                        label[counter*1000:(counter+1)*1000] =\
                        np.argmax(output[counter*1000:(counter+1)*1000],1)
                    hidden_x1 = net.scores(X1)
                    hidden_x2 = net.scores(X2)
                    hidden_x3 = net.scores(X3)
                    bounds= [np.min(hidden[:,0]),np.max(hidden[:,0]),
                             np.min(hidden[:,1]),np.max(hidden[:,1])]
                    color_label=np.zeros_like(colors)
                    color_label[:,0]=(label==0)*(output[:,0]/0.1)//1*.1
                    color_label[:,1]=(label==1)*(output[:,1]/0.1)//1*.1
                    color_label[:,2]=(label==2)*(output[:,2]/0.1)//1*.1
                    rcParams['figure.figsize'] = 20, 10
                    rcParams.update({'font.size': 15})    
                    plt.subplot(1,2,1)
                    plt.scatter(points[:,0],points[:,1],marker='s',s=40,facecolor=colors,
                                edgecolor = '',lw=0)
                #     plt.scatter(X1[:,0],X1[:,1],marker='*',s=10,c='r',edgecolor = '',lw=0)
                #     plt.scatter(X2[:,0],X2[:,1],marker='o',s=10,c='b',edgecolor = '',lw=0)
                #     plt.scatter(X3[:,0],X3[:,1],marker='s',s=10,c='m',edgecolor = '',lw=0)
                    plt.subplot(1,2,2)
                    plt.scatter(hidden[:,0],hidden[:,1],marker='s',s=30,facecolor=colors,
                                edgecolor = '',lw=0)
                    plt.axis(bounds)
                    plt.savefig("pictures_deform_training/run{}_s{}_{}layer_final_epoch{}.png".format(number,shomar,
                                                                                                      num_layer,epoch))
                    
                
                    #plt.scatter(hidden_x1[:,0],hidden_x1[:,1],marker='o',s=15,c='r',
                                #edgecolor = 'w',lw=0.5)
                    #plt.scatter(hidden_x2[:,0],hidden_x2[:,1],marker='o',s=15,c='b',
                                #edgecolor = 'w',lw=0.5)
                    #plt.scatter(hidden_x3[:,0],hidden_x3[:,1],marker='o',s=15,c='m',
                                #edgecolor = 'w',lw=0.5)
                    plt.title("{} size 10 hidden layers + one size 2 hidden layer".format(num_layer-1))
                    #plt.savefig("pictures_deform_training/run{}_s{}_{}layer_final_epoch{}_datapoints.png".format(number,
                                                                                                                 #shomar,num_layer,epoch))
                                #bbox_inches='tight')
                    net.optimize((X_train,y_train), (X_val,y_val), epochs=1,
                                 load = True, save = True, verbose=True, save_always=True)
