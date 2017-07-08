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
import scipy
import scipy.stats as stats
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

class CNN_centers(object):
    
    def __init__(self,  num_centers, centers_dist, network_name,
                 num_conv_layers, num_forward_layers, input_shape,
                 num_classes, path,kernel_sizes, hidden_sizes, 
                 pool_sizes, dims, learning_rate = 0.001, 
                 padding = "SAME", initialize=False,dropout = 1, 
                 reject_cost = 0.2, activation="relu",reg = 0,
                 dynamic = False, batch_norm=True, loss_coeff=0):
        self.network_name = network_name
        self.centers_dist=centers_dist
        self.num_forward_layers = num_forward_layers
        self.num_conv_layers = num_conv_layers
        self.batch_norm = batch_norm
        self.input_shape = input_shape 
        #FIXIT: assumption is images are square
        self.dims = dims
        self.loss_coeff=loss_coeff
        self.padding = padding
        self.learning_rate = learning_rate
        self.pool_sizes = pool_sizes
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.num_classed = num_classes
        self.dropout = dropout
        self.dynamic = dynamic
        self.num_classes = num_classes
        self.path = path
        self.reg = reg
        self.flatten_size = self.flatten_size_calculator()
        self.Pdic = self.make_Pdic()
        self.initialize = initialize
        self.num_centers=num_centers
        self.dropout_ph = tf.placeholder(tf.float32)
        self.input_ph = tf.placeholder\
        (dtype= tf.float32,shape= [None, self.input_shape[0],self.\
                                   input_shape[1],self.input_shape[2]])
        self.output_ph = tf.placeholder(dtype= tf.int32, shape= [None,])
        self.is_training_ph = tf.placeholder(tf.bool)
        self.activation = self.get_activation(activation)
        self.build(self.input_ph)
        
    def make_Pdic(self):
        
        Pdic = {}
        
        Pdic["W"] = tf.get_variable\
        ("W", shape = [self.hidden_sizes[self.num_forward_layers-1]\
                       ,self.num_classes],initializer=\
         tf.contrib.layers.xavier_initializer())
        Pdic["b"] = tf.get_variable("b", shape=[self.num_classes],\
                                    initializer=tf.zeros_initializer())
        
        self.sum_weights = tf.reduce_sum( Pdic["W"]**2)
        
        flat_length = self.flatten_size[self.num_conv_layers-1]
        
        for number in range(self.num_conv_layers):
            Pdic["K{}".format(number)] =  tf.get_variable\
            ("K{}".format(number),shape=[self.kernel_sizes[number]\
                                         ,self.kernel_sizes[number]\
                                         ,(number==0)*self.input_shape[-1]\
                                         + (number>0)*self.dims[number-1],\
                                         self.dims[number]],initializer=\
             tf.contrib.layers.xavier_initializer())
            Pdic["z{}".format(number)] = tf.get_variable\
            ("z{}".format(number), shape = [self.dims[number]],\
             initializer=tf.zeros_initializer())
        
        for layer in range(self.num_forward_layers):
            
            Pdic["W{}".format(layer)] = tf.get_variable\
            ("W{}".format(layer),shape=[flat_length*(layer==0)+\
                                        self.hidden_sizes[layer-1]*(layer>0)\
                                        ,self.hidden_sizes[layer]],
                                 initializer=\
             tf.contrib.layers.xavier_initializer())
            self.sum_weights += tf.reduce_sum( Pdic["W{}".format(layer)]**2)
            Pdic["b{}".format(layer)] = tf.get_variable\
            ("b{}".format(layer),shape=[self.hidden_sizes[layer]]\
             , initializer=tf.zeros_initializer())
        return Pdic
        


    def get_activation(self, name):
        if name == "relu":
            return tf.nn.relu
        if name == "tanh":
            return tf.nn.tanh
        if name == "sigmoid":
            return tf.sigmoid
    
    
    def conv_layer(self, number, feed):
        
        conv = tf.nn.conv2d(input=feed, filter=self.Pdic["K{}".format(number)]\
                            , padding="SAME", strides=[1,1,1,1])
        out_convv =\
        self.activation(conv + self.Pdic["z{}".format(number)])
        if self.batch_norm:
            out_conv = tf.layers.batch_normalization\
            (out_convv,axis=-1,training=self.is_training_ph)
        else:
            out_conv = out_convv
        pool = tf.layers.max_pooling2d\
        (inputs=out_conv, pool_size=self.pool_sizes[number],\
         strides=self.pool_sizes[number])
        return pool
    
    def fc_layer(self, layer, feed):
        out = tf.matmul(feed,self.Pdic["W{}".format(layer)])+\
        self.Pdic["b{}".format(layer)]
        out_relued = tf.nn.dropout(self.activation(out), 
                                   self.dropout_ph)
        return out_relued
    
    def flatten_size_calculator(self):
        if self.num_conv_layers:
            output = np.zeros(self.num_conv_layers)
            temp = self.input_shape[0]//self.pool_sizes[0]
            output[0] = temp*temp*self.dims[0]
            for n in range(1,self.num_conv_layers):
                temp = temp//self.pool_sizes[n]
                output[n] = temp*temp*self.dims[n]
        else:
            output = np.array([self.input_shape[0]*self.input_shape[1]*\
            self.input_shape[2]])
        return output.astype(int)
    
        
    def build(self, feed):
        # FIXIT: assumption is all convolutions are square
        out = feed
        for layer in range(self.num_conv_layers):
            out = self.conv_layer(layer, out)
        flat_length = self.flatten_size[self.num_conv_layers-1]
        out = tf.reshape(out, shape=[-1, flat_length])
        for layer in range(self.num_forward_layers):
            out = self.fc_layer(layer,out)
        self.hidden = out
        self.dic = {}
        self.centers_init=self.centers_dist*\
        np.random.random((self.num_classes,self.num_centers,
                         self.hidden_sizes[-1]))
        self.centers_var=tf.cast(tf.Variable(self.centers_init),
                                 tf.float32)   
        labels = tf.cast(tf.one_hot(self.output_ph,self.num_classes),\
                         tf.float32)
        self.distances = tf_l2_distance_expert(self.hidden,
                                               self.centers_var)

        self.min_distance = tf.reduce_min(self.distances,axis=-1)
        self.which_center = tf.argmin(self.distances, axis=-1)
        self.dic["cost_h"] =\
        tf.reduce_mean(tf.reduce_sum(self.min_distance * labels,axis=1))-\
        self.loss_coeff*tf.reduce_sum(tf_self_distances\
        (tf.reshape(self.centers_var,[-1,self.hidden_sizes[-1]])))\
        +0.5*self.reg*(self.sum_weights)
        
        
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):                                                                       
            self.dic["optmz_h"]= tf.train.AdamOptimizer\
            (self.learning_rate).minimize(self.dic["cost_h"])
            
    def get_batches(self, training_data, batch_size, validation_data):   
        Xs = training_data[0]
        Ys = training_data[1]
        mask = np.random.permutation(len(Ys))
        Xs = Xs[mask]
        Ys = Ys[mask]
        X_batches = [Xs[k:k + batch_size] for k in range(0, len(Xs)\
                                                         , batch_size)]
        Y_batches = [Ys[k:k + batch_size] for k in range(0, len(Ys)\
                                                         , batch_size)]
        return X_batches, Y_batches, validation_data[0], validation_data[1]


    def do_epoch(self, sess, epoch, X_batches, Y_batches, X_val, Y_val,
                 X_train, Y_train, verbose):
        avg_cost = 0
        mskn = np.random.choice(range(X_train.shape[0]),len(X_val))
        x_train = X_train[mskn]
        y_train = Y_train[mskn]
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            sess.run(self.dic["optmz_h"],
                     feed_dict={self.input_ph:X_batch,
                                self.output_ph:Y_batch,
                                self.is_training_ph:True,
                               self.dropout_ph:self.dropout})
        val_h = sess.run(self.hidden,
                         feed_dict={self.input_ph:X_val,
                                    self.is_training_ph:False,
                                    self.dropout_ph:1})
        val_cost = sess.run(self.dic["cost_h"],
                            feed_dict={self.input_ph:X_val,
                                       self.output_ph:Y_val,
                                       self.is_training_ph:False,
                                       self.dropout_ph:1})
        acc, rate = self.evaluate(val_h,Y_val)
        tr_h = sess.run(self.hidden,
                        feed_dict={self.input_ph:x_train,
                                   self.is_training_ph:False,
                                   self.dropout_ph:1})
        tr_cost = sess.run(self.dic["cost_h"],
                           feed_dict={self.input_ph:x_train,
                                      self.output_ph:y_train,
                                      self.is_training_ph:False,
                                      self.dropout_ph:1})
        acc_train, _ = self.evaluate(tr_h,y_train)
        if verbose:
            print("Epoch:{}".format(epoch))
            print("Val/Train Accuracy:{}/{}".format(acc,acc_train))
            print("Val/Train Cost:{}/{}".format(val_cost,tr_cost))
            if rate<1:
                print("rate:{}").format(rate)
        return acc
    
    def tradeoff(self,sess, X_val, Y_val):
        self.acc_hist = []
        self.rate_hist = []
        self.thresh_list = []
        val_h = sess.run(self.hidden,\
                         feed_dict={self.input_ph:X_val,\
                                    self.is_training_ph:False,
                                    self.dropout_ph:1})
        r=0.
        thresh = 1e5*1.
        while(r<0.95):
            thresh /= 1.001
            self.thresh_list.append(thresh)
            a,r = self.evaluate(val_h,Y_val,thresh)
            r = 1-r
            self.acc_hist.append(a)
            self.rate_hist.append(r)
#                 print(thresh,a,r)
          
    def give_back_centers(self,X_train,y_train):
        orig_centers=np.zeros((self.num_classes,self.num_centers,
                              self.input_shape[0],self.input_shape[1],
                              self.input_shape[2]))
        for i in range(self.num_classes):
            args = np.where(y_train==i)[0]
            args_center = np.random.choice(args,self.num_centers)
            orig_centers[i]=X_train[args_center]
        orig_centers=np.reshape(orig_centers,[self.num_classes*self.num_centers,
                                              self.input_shape[0],self.input_shape[1],
                                               self.input_shape[2]])
        return orig_centers
    
    def optimize(self,  training_data, validation_data,
                 save, load , reg= 0, epochs=10,batch_size= 200, 
                 tradeoff=False, verbose=True, save_always=False, loss_coeff=0):
        
        X_batches, Y_batches, X_val, Y_val = \
        self.get_batches(training_data, batch_size, validation_data)
        X_train = training_data[0]
        Y_train = training_data[1]
        self.orig_centers=self.give_back_centers(X_train,Y_train)
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        
            
        with tf.Session(config=config_gpu) as sess:
            sess.run(init)
            best_val = 0.
#             if self.num_centers>1:
#                 initilization_centers = \
#                 tf.reshape(self.feedforward(self.orig_centers,sess),
#                            [self.num_classes,self.num_centers,
#                                   self.hidden_sizes[-1]])
#             else:
#                 initilization_centers =\
#                 self.feedforward(self.orig_centers,sess)
             
            self.centers=sess.run(self.centers_var)   
            if load:
                if verbose:
                    print("Loading model from :{}".format(self.path))
                saver.restore(sess,self.path)
            self.centers,val_h = sess.run([self.centers_var,self.hidden],\
                             feed_dict={self.input_ph:X_val,\
                                        self.is_training_ph:False,
                                        self.dropout_ph:1})
            best_val, _ = self.evaluate(val_h,Y_val)
            if verbose:
                print("validation accuracy before starting",best_val)

            for epoch in range(epochs):
                print(np.linalg.norm\
                      (self.centers-sess.run(self.centers_var)))
                self.centers=sess.run(self.centers_var)
                acc = self.do_epoch(sess, epoch, X_batches, Y_batches,
                                    X_val, Y_val,X_train,
                                    Y_train, verbose)
                if acc>=best_val or save_always: #FIXIT
                    best_val = acc
                    saved_path = saver.save(sess, self.path)
                    if verbose and not save_always:
                        print("New best!")
            self.best_val = best_val 
            print("Best val accuracy:",self.best_val)
            saver.restore(sess,self.path)
            if tradeoff:
                self.tradeoff(sess, X_val, Y_val)

    def predict(self, hidden):
        index = -1
        counter = 0.
        correct = 0.
        out = np.zeros(hidden.shape[0])
        diff_tensor =\
        tf.minimum(tf_l2_distance_expert(tf.constant(hidden)
                                         ,self.centers_var),axis=-1)
        diff=sess.run(diff_tensor)
        out = np.argmin(diff,axis=1)
        return out
    
    def adversary(self, feed, epsilon, true_label, mode="BIM",
                  sess=None,max_iter=10):
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)
        net_label=np.argmax(self.feedforward(feed,sess),1)
        if net_label!=true_label:
            print("Network is already misclassifying")
            return
        adv = feed
        iters=0
        if mode=="BIM":
            cont=True
            while(cont and iters<max_iter):
                iters += 1
                gradient = sess.run(tf.gradients(self.dic["cost_h"],
                                        self.input_ph)[0],
                                    feed_dict={self.input_ph:adv,
                                   self.is_training_ph:False,
                                   self.output_ph:true_label,
                                   self.dropout_ph:1})
                adv = feed+np.clip(adv+epsilon*np.sign(gradient)-feed,
                                  -epsilon,epsilon)
                out_adv = self.feedforward(adv,sess)
                print("True label:",true_label,"New label:",
                     np.argmax(out_adv,1),"Confidence:",
                     np.max(out_adv,1))
                if np.argmax(out_adv,1)!=\
                true_label and np.max(out_adv,1)>0.95:
                    cont=False
        return adv
            
    def evaluate(self, hidden, y, thresh=1e100):  
        correct = 0.
        if self.num_centers>1:
            diff = np.min(l2_distance_expert(hidden,self.centers),
                          axis=-1)
        else:
            diff = l2_distance(hidden,self.centers)
        passed = np.min(diff,1)<thresh
        passed_num = np.sum(passed)
        correct = np.argmin(diff,1) == y     
        return np.sum(correct*passed)*1./passed_num,\
    passed_num*1./y.shape[0]
    
    def feedforward(self, feed, sess=None):
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        output = sess.run\
        (self.hidden, {self.input_ph: feed,
                       self.is_training_ph : False,
                       self.dropout_ph:1})
    
    
class CNN_SC(object):
    
    def __init__(self, network_name, num_conv_layers\
                 , num_forward_layers, input_shape, num_classes, path\
                 ,kernel_sizes, hidden_sizes, pool_sizes, dims,\
                 learning_rate = 0.001, padding = "SAME", initialize=False,
                 dropout = 1, reject_cost = 0.2, activation="relu"\
                 ,reg = 0, dynamic = False, batch_norm=True, double=False):
        self.network_name = network_name
        self.num_forward_layers = num_forward_layers
        self.num_conv_layers = num_conv_layers
        self.batch_norm = batch_norm
        self.input_shape = input_shape 
        #FIXIT: assumption is images are square
        self.dims = dims
        self.padding = padding
        self.learning_rate = learning_rate
        self.pool_sizes = pool_sizes
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.num_classed = num_classes
        self.dropout = dropout
        self.num_classes = num_classes
        self.path = path
        self.cost_history = []
        self.reg = reg
        self.out_dict = {}
        self.flatten_size = self.flatten_size_calculator()
        self.Pdic = self.make_Pdic()
        self.initialize = initialize
        self.dropout_ph = tf.placeholder(tf.float32)
        self.input_ph = tf.placeholder\
        (dtype= tf.float32,shape= [None, self.input_shape[0],self.\
                                   input_shape[1],self.input_shape[2]])
        self.output_ph = tf.placeholder(dtype= tf.int32, shape= [None,])
        self.is_training_ph = tf.placeholder(tf.bool)
        self.activation = self.get_activation(activation)
        self.build(self.input_ph)
        
    def make_Pdic(self):
        
        Pdic = {}
        
        Pdic["W"] = tf.get_variable\
        ("W", shape = [self.hidden_sizes[self.num_forward_layers-1]\
                       ,self.num_classes],initializer=\
         tf.contrib.layers.xavier_initializer())
        Pdic["b"] = tf.get_variable("b", shape=[self.num_classes],\
                                    initializer=tf.zeros_initializer())
        
        self.sum_weights = tf.reduce_sum( Pdic["W"]**2)
        
        flat_length = self.flatten_size[self.num_conv_layers-1]
        
        for number in range(self.num_conv_layers):
            Pdic["K{}".format(number)] =  tf.get_variable\
            ("K{}".format(number),shape=[self.kernel_sizes[number]\
                                         ,self.kernel_sizes[number]\
                                         ,(number==0)*self.input_shape[-1]\
                                         + (number>0)*self.dims[number-1],\
                                         self.dims[number]],initializer=\
             tf.contrib.layers.xavier_initializer())
            Pdic["z{}".format(number)] = tf.get_variable\
            ("z{}".format(number), shape = [self.dims[number]],\
             initializer=tf.zeros_initializer())
        
        for layer in range(self.num_forward_layers):
            
            Pdic["W{}".format(layer)] = tf.get_variable\
            ("W{}".format(layer),shape=[flat_length*(layer==0)+\
                                        self.hidden_sizes[layer-1]*(layer>0)\
                                        ,self.hidden_sizes[layer]],
                                 initializer=\
             tf.contrib.layers.xavier_initializer())
            self.sum_weights += tf.reduce_sum( Pdic["W{}".format(layer)]**2)
            Pdic["b{}".format(layer)] = tf.get_variable\
            ("b{}".format(layer),shape=[self.hidden_sizes[layer]]\
             , initializer=tf.zeros_initializer())
        return Pdic
        

    
    def calculate_dic(self):
        dic = {}
        corrects = tf.cast(tf.equal(tf.cast(tf.argmax(self.out,1),tf.int32),\
                                    self.output_ph),"float")
        dic["accuracy"] = tf.reduce_mean(corrects)  
        return dic

    def get_activation(self, name):
        if name == "relu":
            return tf.nn.relu
        if name == "tanh":
            return tf.nn.tanh
        if name == "sigmoid":
            return tf.sigmoid
    
    
    def conv_layer(self, number, feed):
        
        conv = tf.nn.conv2d(input=feed, filter=self.Pdic["K{}".format(number)]\
                            , padding="SAME", strides=[1,1,1,1])
        out_convv = self.activation(conv + self.Pdic["z{}".format(number)])
        if self.batch_norm:
            out_conv = tf.layers.batch_normalization\
            (out_convv,axis=-1,training=self.is_training_ph)
        else:
            out_conv = out_convv
        pool = tf.layers.max_pooling2d\
        (inputs=out_conv, pool_size=self.pool_sizes[number],\
         strides=self.pool_sizes[number])
        return pool
    
    def fc_layer(self, layer, feed):
        out = tf.matmul(feed,self.Pdic["W{}".format(layer)])+\
        self.Pdic["b{}".format(layer)]
        out_relued = tf.nn.dropout(self.activation(out), 
                                   self.dropout_ph)
        return out_relued
    
    def flatten_size_calculator(self):
        if self.num_conv_layers:
            output = np.zeros(self.num_conv_layers)
            temp = self.input_shape[0]//self.pool_sizes[0]
            output[0] = temp*temp*self.dims[0]
            for n in range(1,self.num_conv_layers):
                temp = temp//self.pool_sizes[n]
                output[n] = temp*temp*self.dims[n]
        else:
            output = np.array([self.input_shape[0]*self.input_shape[1]*\
            self.input_shape[2]])
        return output.astype(int)
    
        
    def build(self, feed):
        out = feed
        for layer in range(self.num_conv_layers):
            out = self.conv_layer(layer, out)
        flat_length = self.flatten_size[self.num_conv_layers-1]
        out = tf.reshape(out, shape=[-1, flat_length])
        for layer in range(self.num_forward_layers):
            out = self.fc_layer(layer,out)
        self.hidden = out
        self.out = tf.matmul(self.hidden,self.Pdic["W"])+self.Pdic["b"]
        self.dic = self.calculate_dic()
        labels = tf.cast(tf.one_hot(self.output_ph,self.num_classes),\
                         tf.float32)
        self.dic["cost"] = tf.reduce_mean\
        (tf.nn.sparse_softmax_cross_entropy_with_logits\
        (logits=self.out, labels=self.output_ph))\
        + 0.5*self.reg*(self.sum_weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):            
            self.dic["optmz"]= tf.train.AdamOptimizer\
            (self.learning_rate).minimize(self.dic["cost"])
        if self.initialize:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer() 
            sess = get_session()
            sess.run(init)
            saver.save(sess,self.path)
        
    def get_batches(self, training_data, batch_size, validation_data):   
        Xs = training_data[0]
        Ys = training_data[1]
        mask = np.random.permutation(len(Ys))
        Xs = Xs[mask]
        Ys = Ys[mask]
        X_batches = [Xs[k:k + batch_size] for k in range(0, len(Xs)\
                                                         , batch_size)]
        Y_batches = [Ys[k:k + batch_size] for k in range(0, len(Ys)\
                                                         , batch_size)]
        return X_batches, Y_batches, validation_data[0], validation_data[1]


    def do_epoch(self, sess, epoch, X_batches, Y_batches, X_val, Y_val,
                 X_train, Y_train, verbose):
        mskn = np.random.choice(range(X_train.shape[0]),len(X_val))
        x_train = X_train[mskn]
        y_train = Y_train[mskn]
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            sess.run([self.dic["optmz"], self.dic["cost"]],
                     feed_dict={self.input_ph:X_batch,
                                self.output_ph:Y_batch,
                                self.is_training_ph:True,
                                self.dropout_ph:self.dropout})
        acc = sess.run(self.dic["accuracy"],
                       feed_dict={self.input_ph:X_val,
                                  self.output_ph:Y_val,
                                  self.is_training_ph:False,
                                  self.dropout_ph:1})
        acc_train = sess.run(self.dic["accuracy"],
                             feed_dict={self.input_ph:x_train,
                                        self.output_ph:y_train,
                                        self.is_training_ph:False,
                                        self.dropout_ph:1})
        tr_cost = sess.run(self.dic["cost"],
                           feed_dict={self.input_ph:x_train,
                                      self.output_ph:y_train,
                                      self.is_training_ph:False,
                                      self.dropout_ph:1})
        val_cost = sess.run(self.dic["cost"], 
                          feed_dict={self.input_ph:X_val,
                                    self.output_ph:Y_val,
                                    self.is_training_ph:False,
                                     self.dropout_ph:1})
        if verbose:
            print("Epoch:{}".format(epoch))
            print("Val/Train Accuracy:{}/{}".format(acc,acc_train))
            print("Val/Train Cost:{}/{}".format(val_cost,tr_cost))
        return acc
    
    def tradeoff(self,sess, X_val, Y_val):
        self.acc_hist = []
        self.rate_hist = []
        self.thresh_list = []
        val = sess.run(tf.nn.softmax(self.out),\
                       feed_dict={self.input_ph:X_val, \
                                  self.is_training_ph:False,
                                  self.dropout_ph:1})
        ent = entropy(val)
        r = 0.
        thresh = 10.
        while(r<.95):
            thresh /= 1.01
            self.thresh_list.append(thresh)
            passes = (ent<thresh).astype(float)
            corrects = (np.argmax(val,1)==Y_val).astype(float) * passes
            a = np.sum(corrects)/np.sum(passes)
            r = 1-np.sum(passes)/val.shape[0]
            self.acc_hist.append(a)
            self.rate_hist.append(r)
#                 print(thresh,a,r)
        
    def optimize(self, training_data, validation_data,
                 save, load , reg= 0, epochs=10,batch_size= 200, 
                 tradeoff=False, verbose=True, save_always=False):
        X_batches, Y_batches, X_val, Y_val = \
        self.get_batches(training_data, batch_size, validation_data)
        X_train = training_data[0]
        Y_train = training_data[1]
        saver = tf.train.Saver()
        init = tf.global_variables_initializer() 

        with tf.Session(config=config_gpu) as sess:
            sess.run(init)
            best_val = 0.
            
            if load:
                if verbose:
                    print("Loading model from :{}".format(self.path))
                saver.restore(sess,self.path)
            best_val = sess.run(self.dic["accuracy"], 
                                feed_dict={self.input_ph:X_val,
                                           self.output_ph:Y_val,
                                           self.is_training_ph:False,
                                           self.dropout_ph:1})
            if verbose:
                print("validation accuracy before starting",best_val)
                
            for epoch in range(epochs):
                acc = self.do_epoch(sess, epoch, X_batches, Y_batches,
                                    X_val, Y_val, X_train,
                                    Y_train, verbose)
                if acc>=best_val or save_always: #FIXIT
                    best_val = acc
                    saved_path = saver.save(sess, self.path)
                    if verbose and not save_always:
                        print("New best!")
            self.best_val = best_val 
            print("Best val accuracy:",self.best_val)
            saver.restore(sess,self.path)
            if tradeoff:
                self.tradeoff(sess, X_val, Y_val)

    def tf_PolMut(self, x, eta=15., prob=0.1,up=255.0,down=0.):
        up = up*1.
        down = down*1.
        eta = eta*1.
        mask = tf.cast(tf.less(tf.random_uniform(tf.shape(x),0,1),prob),
                       tf.float32)
        changed = mask*x
        unchanged = (1-mask)*x
        delta1 = x*mask/(up-down)
        delta2 = (1-x)*mask/(up-down)
        r = tf.cast(tf.random_uniform(tf.shape(x),0,1),tf.float32)
        r_mask = tf.cast(tf.less(r,0.5),tf.float32)
        r_mask1 = r_mask * mask
        r_mask2 = (1-r_mask) * mask
        r1 = r * r_mask1
        r2 = r * r_mask2
        q1 = (2*r1+(1-2*r1)*r_mask1*(mask*(1-delta1))**(eta+1))
        q2 = (2*(1-r2)+2*(r2-0.5)*r_mask2*(mask*(1-delta2))**(eta+1))
        d = (q1**(1./(eta+1))-1) * r_mask1 + (1-q2**(1./(eta+1))) * r_mask2
        result = unchanged + changed+d*(up-down)
        return result
    
    def PM(self, num_pic, num_iter, num_sample):
        saver = tf.train.Saver()
        with tf.Session(config=config_gpu) as sess:
            saver.restore(sess,self.path)
            best_pic = np.zeros((10*num_pic,32,32,3))
            best_score = 1e100*np.ones(10*num_pic)*1.
            
            for ctr in range(num_pic):
                x = 255*np.random.random((num_sample,32,32,3))    
                print(ctr)
                for i in range(num_iter):
                    out = sess.run(tf.nn.softmax(self.out),
                                   {self.input_ph: x-mean_image,
                                    self.is_training_ph: False,
                                    self.dropout_ph:1})
                    max_index = out.argmax(axis=0)
                    for cls in range(10):
                        if out[max_index[cls]][cls]>\
                        best_score[cls*num_pic+ctr]:
                            best_score[cls*num_pic+ctr] =\
                            out[max_index[cls]][cls]
                            best_pic[cls*num_pic+ctr] =\
                            x[max_index[cls]]-mean_image
                    x = sess.run\
                    (self.tf_PolMut\
                     (tf.constant(x,dtype=tf.float32),15,.1,255.,0.))
        return best_pic, best_score
    
    def hist_dist(self,x,y):
        saver = tf.train.Saver()
        with tf.Session(config=config_gpu) as sess:
            saver.restore(sess,self.path)
            if self.double:
                distances1 = sess.run\
                (self.distances1,{self.input_ph: x, self.output_ph:y,
                  self.is_training_ph : False, self.dropout_ph:1})
                distances2 = sess.run\
                (self.distances2, {self.input_ph: x, self.output_ph:y,
                                   self.is_training_ph : False,
                                   self.dropout_ph:1})
                distances = np.minimum(distances1,distances2)
            else:
                distances = sess.run\
                (self.distances, {self.input_ph: x, self.output_ph:y,
                                  self.is_training_ph : False,
                                  self.dropout_ph:1})
            output1 = distances[np.arange(y.shape[0]),y]
            return output1,distances
    
    def adversary(self, feed, mean_image, epsilon, true_label, 
                  mode="BIM",sess=None,max_iter=30,verbose=False,
                  alpha=1):
        feed=np.expand_dims(feed,axis=0)
        true_label=[true_label]
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)
        net_probs = self.feedforward(feed,sess)
        net_label=np.argmax(net_probs,1)
        fake_label=np.argmin(net_probs,1)
        if net_label!=true_label:
            if verbose:
                print("Network is already misclassifying")
            return 
        adv = feed
        iters=0
        cont=True
        changed_label=False
        while(cont and iters<max_iter):
            iters += 1
            if mode=="sign":
                gradient = sess.run(tf.gradients(self.dic["cost"],
                                        self.input_ph)[0],
                                    feed_dict={self.input_ph:adv,
                                   self.is_training_ph:False,
                                   self.output_ph:true_label,
                                   self.dropout_ph:1})
                adv_temp = adv + epsilon*np.sign(gradient)
                cont=False
            if mode=="BIM":
                gradient = sess.run(tf.gradients(self.dic["cost"],
                                        self.input_ph)[0],
                                    feed_dict={self.input_ph:adv,
                                   self.is_training_ph:False,
                                   self.output_ph:true_label,
                                   self.dropout_ph:1})
                adv_temp = adv + alpha*np.sign(gradient)
            if mode=="ILLCM":
                gradient = sess.run(tf.gradients(self.dic["cost"],
                                        self.input_ph)[0],
                                    feed_dict={self.input_ph:adv,
                                   self.is_training_ph:False,
                                   self.output_ph:fake_label,
                                   self.dropout_ph:1})
                adv_temp = adv - alpha*np.sign(gradient)
            adv_temp=np.maximum(np.maximum(feed-epsilon,adv_temp),
                             -mean_image)
            
            adv=np.minimum(np.minimum(feed+epsilon,adv_temp),
                           255-mean_image)
            
            out_adv = self.feedforward(adv,sess)
            confidence = np.max(out_adv,1)
            new_label = np.argmax(out_adv,1)
            if verbose:
                print("True label:",true_label,"New label:",
                     new_label,"Confidence:", confidence)
            if new_label!=true_label:
                changed_label=True
            if changed_label and np.max(out_adv,1)>0.99:
                cont=False
        return adv, changed_label, confidence
            
    
    def feedforward(self, feed, sess=None):
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)            
        output = sess.run\
        (tf.nn.softmax(self.out), {self.input_ph: feed,
                                   self.is_training_ph : False,
                                   self.dropout_ph:1})
        return output
    
    def scores(self, feed, sess=None):
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        output = sess.run\
        (self.hidden, {self.input_ph: feed, self.is_training_ph : False,
                       self.dropout_ph:1})
        return output