from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.misc import imread
import platform


def l2_distance(a,b):
    norm_a = np.sum(a**2,1,keepdims=True)
    norm_b = np.sum((b.T)**2,0,keepdims=True)
    output = -2*a.dot(b.T)
    output += norm_a
    output += norm_b
    return np.sqrt(output)


def tf_l2_distance(a,b):
    norm_a = tf.reduce_sum(a**2,axis=1,keep_dims=True)
    norm_b = tf.reduce_sum(tf.transpose(b)**2,axis=0,keep_dims=True)
    output = -2*tf.matmul(a,tf.transpose(b))
    output += norm_a
    output += norm_b
    return tf.sqrt(output)


def analyze_distance(X_val,X_train,reset_graph=True,sess=None,h=32,w=32,c=3):
    if reset_graph:
        tf.reset_default_graph()
    if sess==None:
        sess=get_session()
    a=tf.constant(X_val)
    length = len(X_train)//1000
    length_val = len(X_val)//1000
    val_reshaped = tf.reshape(a,[-1,w*h*c])
    dist_train = np.zeros((length_val*1000,length*1000))
    for i in range(length):
        for j in range(length_val):
            dist_train[j*1000:(j+1)*1000,i*1000:(i+1)*1000] =\
            sess.run(tf_l2_distance\
                     (tf.reshape\
                      (tf.constant(X_val[j*1000:(j+1)*1000]),
                       [-1,w*h*c]),tf.reshape\
                      (tf.constant(X_train[i*1000:(i+1)*1000]),
                       [-1,w*h*c])))
    args_train = np.argsort(dist_train,1)
    sorted_train = np.sort(dist_train,1)
    return{"dist":dist_train,"args":args_train,
           "sorted":sorted_train}


def tf_l2_distance_expert(a,b):
    num_classes=b.get_shape().as_list()[0]
    num_centers=b.get_shape().as_list()[1]
    hidden_dim = b.get_shape().as_list()[2]
    b = tf.reshape(b,[-1,hidden_dim])
    norm_a = tf.reduce_sum(a**2,axis=1,keep_dims=True)
    norm_b = tf.reduce_sum(tf.transpose(b)**2,axis=0,keep_dims=True)
    output = -2*tf.matmul(a,tf.transpose(b))
    output += norm_a
    output += norm_b
    out = tf.sqrt(output)
    return tf.reshape(output,[-1,num_classes,num_centers])


def l2_distance_expert(a,b):
    num_centers=1
    num_centers=b.shape[1]
    num_classes=b.shape[0]
    b = np.reshape(b,[-1,b.shape[-1]])
    norm_a = np.sum(a**2,axis=1,keepdims=True)
    norm_b = np.sum((b.T)**2,axis=0,keepdims=True)
    output = -2*np.dot(a,(b.T)) + norm_a + norm_b
    out = np.sqrt(output)
    return np.reshape(output,[-1,num_classes,num_centers]) 
       
    
def tf_self_distances(centers,mode="sum"):
    
    num_classes = centers.get_shape().as_list()[0]
    num_centers = centers.get_shape().as_list()[1]
    hidden_dim = centers.get_shape().as_list()[2]
    out=None
    for counter1 in range(num_classes):
        for counter2 in range(counter1+1,num_classes):
            a = centers[counter1]
            b = centers[counter2]
            norm_a = tf.reduce_sum(a**2,axis=1,keep_dims=True)
            norm_b = tf.reduce_sum(tf.transpose(b)**2,axis=0,
                                   keep_dims=True)
            output = -2*tf.matmul(a,tf.transpose(b))
            output += norm_a
            output += norm_b
            if mode=="sum":
                if out==None:
                    out = tf.reduce_sum(output)
                else:
                    out += tf.reduce_sum(output)
            elif mode=="min":
                if out==None:
                    out = tf.reduce_sum(output)
                else:
                    out += tf.reduce_sum(output)
            else:
                raise ValueError("Wrong mode!")
    return out

    
    
def analyze_distance(X_val,X_train,reset_graph=True,sess=None,h=32,w=32,c=3):
    if reset_graph:
        tf.reset_default_graph()
    if sess==None:
        sess=get_session()
    a=tf.constant(X_val)
    length = len(X_train)//1000
    val_reshaped = tf.reshape(a,[-1,w*h*c])
    dist_train = np.zeros((X_val.shape[0],length*1000))
    for i in range(length):
        dist_train[:,i*1000:(i+1)*1000] =\
        sess.run(tf_l2_distance\
                 (val_reshaped,tf.reshape\
                  (tf.constant(X_train[i*1000:(i+1)*1000]),[-1,w*h*c])))
    args_train = np.argsort(dist_train,1)
    sorted_train = np.sort(dist_train,1)
    return{"dist":dist_train,"args":args_train,
           "sorted":sorted_train}


def analyse_adv(image,adv,X_train,reset_graph=True,sess=None):
    if reset_graph:
        tf.reset_default_graph()
    if sess==None:
        sess=get_session()
    a = tf.constant(np.append(image,adv,axis=0))
    length = len(X_train)//1000
    val_reshaped = tf.reshape(a,[-1,32*32*3])
    dist_train = np.zeros((2,length*1000))
    for i in range(length):
        dist_train[:,i*1000:(i+1)*1000] =\
        sess.run(tf_l2_distance\
                 (val_reshaped,tf.reshape\
                  (tf.constant(X_train[i*1000:(i+1)*1000]),[-1,32*32*3])))
    args_train = np.argsort(dist_train,1)
    sorted_train = np.sort(dist_train,1)
    print("distance between adversary and original image:",
          np.linalg.norm(adv-image))
    return{"dist":dist_train,"args":args_train,
           "sorted":sorted_train}

def save_object(object, filename):
    with open(filename, 'wb') as output:
        pickle.dump(object,output,pickle.HIGHEST_PROTOCOL)
        
def load_object(obj, fielname, encoding=None):
    if encoding==None:
        with open(fielname, 'rb') as input:
            obj= pickle.load(input)
    else:
        with open(fielname, 'rb') as input:
            obj = pickle.load(input,encoding=encoding)
        
def entropy(feed):
    return np.sum(-feed*np.log(feed),axis=-1)

def tf_entropy(feed):
    return tf.reduce_sum(-feed*tf.log(feed),axis=-1)

def histogram_plotter(data,bins,xlabel=""):
    from pylab import rcParams
    weights = 100*np.ones_like(data)/len(data)
    plt.hist(data,bins,weights=weights)
    plt.xlabel(xlabel)
    plt.ylabel("%")
    
def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters
 
    
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
 
    
def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))
 
    
def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def unitpoint(N,w,h,c):
    a = np.random.normal(size=(N,w*h*c))
    b = np.expand_dims(np.sqrt(np.sum(a**2,1)),1)
    return np.reshape(a/b,[N,w,h,c])
  
    
def random_triangle(N,w,h,c,r,mean_image,a=None):
    if a==None:
        a = 255*np.random.random((N,w,h,c))-mean_image
        a = np.reshape(a,[N,-1])
    base = np.reshape(r*unitpoint(N,w,h,c),[N,-1])
    base = base/np.expand_dims(np.linalg.norm(base,axis=1),-1)
    b = a + r*base
    orthog = np.reshape(unitpoint(N,w,h,c),[N,-1])
    orthog = orthog - np.expand_dims(np.diagonal(np.inner(orthog,base)),-1) * base
    orthog = orthog/np.expand_dims(np.linalg.norm(orthog,axis=1),-1)
    orthog = orthog * r * np.sqrt(3)/2
    c = a + r*base/2 + orthog
    return np.reshape(a,[N,w,h,-1]),np.reshape(b,[N,w,h,-1]),np.reshape(c,[N,w,h,-1])


def l2_distance(a,b):
    norm_a = np.sum(a**2,1,keepdims=True)
    norm_b = np.sum((b.T)**2,0,keepdims=True)
    output = -2*a.dot(b.T)
    output += norm_a
    output += norm_b
    return np.sqrt(output)

 
    
def tf_self_distances(a):
    norm=tf.norm(a,axis=1,keep_dims=True)**2
    return 2*norm-2*tf.matmul(a,tf.transpose(a))

def condition_number_dist(sess,r,NET,iters,mean_image,
                          num_points=32*32*3,w=32,h=32,c=3):
    for i in range(iters):
        if i%100==0:
            print(i)
        point = 255*np.random.random((1,w,h,c))-mean_image
        points = point + r*unitpoint(num_points,w,h,c)
        point_mapped = NET.scores(point,sess)
        points_mapped = NET.scores(points,sess)
        dists = np.linalg.norm(points_mapped-point_mapped,axis=1)
        cond_nums.append(max(dists)/min(dists))
    return cond_nums


def sphere_distortion(r,N,n,NET,w=32,h=32,c=3,
                      hidden_size=1024,verbose=False,X_val=None):
    sess=get_session()
    saver = tf.train.Saver()
    saver.restore(sess,NET.path)  
    cond_nums=[]
    dists = np.zeros(n)
    points = np.zeros((n,w,h,c))
    points_mapped = np.zeros((n,hidden_size))
    for i in range(N):
        if verbose:
            if N>10:
                if i%(N//10)==0:
                    print(i)
            else:
                print(i)
        if X_val==None:
            point = 255*np.random.random((1,w,h,c))-mean_image
        else:
            point = np.expand_dims(X_val[np.random.choice(X_val.shape[0])],0)
        point_mapped = NET.scores(point,sess)
        if n<2000:
            points = point + r*unitpoint(n,32,32,3)
            points_mapped = NET.scores(points,sess)
            dists = np.linalg.norm(points_mapped-point_mapped,axis=1)
        else:
            for ctr in range(n//2000):
                points[ctr*2000:(ctr+1)*2000] = point + r*unitpoint(2000,w,h,c)
                points_mapped[ctr*2000:(ctr+1)*2000] = NET.scores(points[ctr*2000:(ctr+1)*2000],sess)
                dists[ctr*2000:(ctr+1)*2000] = sess.run(tf.norm(tf.constant(points_mapped[ctr*2000:(ctr+1)*2000])-
                                                                       tf.cast(tf.constant(point_mapped),tf.float64),1))

        cond_nums.append(max(dists)/min(dists))
    cond_nums = np.array(cond_nums)
    weights = 100*np.ones_like(cond_nums)/len(cond_nums)
    rcParams['figure.figsize'] = 10, 10
    plt.hist(cond_nums,20,weights=weights)
    plt.xlabel("k")
    plt.ylabel("%")
    print(np.mean(cond_nums),np.var(cond_nums))
    return cond_nums


def get_session():
    """Create a session that dynamically allocates memory."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session


def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
    
    
def PolMut(x, eta, prob,up,down):
    up = up*1.
    down = down*1.
    eta = eta*1.
    mask = (np.random.random(x.shape)<prob).astype("float")
    changed = mask*x
    unchanged = (1-mask)*x
    delta1 = x*mask/(up-down)
    delta2 = (1-x)*mask/(up-down)
    r = np.random.random(x.shape)
    r_mask = (r<0.5).astype("float")
    r_mask1 = r_mask * mask
    r_mask2 = (1-r_mask) * mask
    r1 = r * r_mask1
    r2 = r * r_mask2
    q1 = (2*r1+(1-2*r1)*r_mask1*(mask*(1-delta1))**(eta+1))
    q2 = (2*(1-r2)+2*(r2-0.5)*r_mask2*(mask*(1-delta2))**(eta+1))
    d = (q1**(1./(eta+1))-1) * r_mask1 + (1-q2**(1./(eta+1))) * r_mask2
    result = unchanged + changed+d*(up-down)
    return result



def get_CIFAR10_data(directory,num_training=40000, num_validation=10000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = directory
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    y_val2 = y_val[5000:]
    X_val2 = X_val[5000:]
    train_mask = np.argsort(y_train)
    val_mask = np.argsort(y_val2)
    y_train_sorted = y_train[train_mask].copy()
    y_val_sorted = y_val2[val_mask].copy()
    X_train_sorted = X_train[train_mask].copy()
    X_val_sorted = X_val2[val_mask].copy()
    X_train_0 = X_train_sorted[:3900]
    X_train_1 = X_train_sorted[4000:7900]
    X_train_2 = X_train_sorted[8000:11900]
    X_train_3 = X_train_sorted[12000:15900]
    X_train_4 = X_train_sorted[16100:20000]
    X_train_5 = X_train_sorted[20050:23950]
    X_train_6 = X_train_sorted[24000:27900]
    X_train_7 = X_train_sorted[28100:32000]
    X_train_8 = X_train_sorted[32100:36000]
    X_train_9 = X_train_sorted[36100:]
    X_train_sorted= [X_train_0,X_train_1,X_train_2,X_train_3,X_train_4
                     ,X_train_5,X_train_6,X_train_7,X_train_8,X_train_9]
    X_val_0 = X_val_sorted[:400]
    X_val_1 = X_val_sorted[500:900]
    X_val_2 = X_val_sorted[1000:1400]
    X_val_3 = X_val_sorted[1500:1900]
    X_val_4 = X_val_sorted[2000:2400]
    X_val_5 = X_val_sorted[2500:2900]
    X_val_6 = X_val_sorted[3000:3400]
    X_val_7 = X_val_sorted[3500:3900]
    X_val_8 = X_val_sorted[4000:4400]
    X_val_9 = X_val_sorted[4500:4900]
    X_val_sorted = [X_val_0,X_val_1,X_val_2,X_val_3,X_val_4,
           X_val_5,X_val_6,X_val_7,X_val_8,X_val_9]
    data_dic = {"X_train":X_train, "X_val":X_val, "X_test":X_test,
               "y_train":y_train, "y_val":y_val, "y_test":y_test,
                "mean_image":mean_image}
    aux_dic = {"X_train1":X_train[:25000],
                "y_train1":y_train[:25000],"X_train2":X_train[25000:],
                "y_train2":y_train[25000:], "X_val1":X_val[:2000],
                "y_val1":y_val[:2000],"X_val2":X_val[5000:],
                "y_val2":y_val[5000:],"X_train_sorted":X_train_sorted,
                "X_val_sorted":X_val_sorted}
    return data_dic, aux_dic


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte




def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d'
                  % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
                        np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
        ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
      'class_names': class_names,
      'X_train': X_train,
      'y_train': y_train,
      'X_val': X_val,
      'y_val': y_val,
      'X_test': X_test,
      'y_test': y_test,
      'class_names': class_names,
      'mean_image': mean_image,
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models
