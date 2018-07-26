import math
import numpy as np
import os.path
import urllib.request as urllib
import gzip
import pickle
import pandas as pd

from scipy.misc import imsave
from src.utils.download import *
import src.utils.image_load_helpers as image_load_helpers
import glob

def CelebA_load(label_data = None, image_paths = None, batch_size = 64, isTrain=True):
    path='./data/CelebA/'

    assert os.path.exists(path + 'is_male.csv')
    assert os.path.isdir(path + 'images/')
    
    if( label_data is None ):
        label_data = np.squeeze((pd.read_csv(path + 'is_male.csv') + 1).astype(np.bool).astype(np.int32).values)
        image_paths = glob.glob(path + 'images/*')
        image_paths.sort()
        return 1 - label_data, image_paths
    
    tot_len = len(label_data)
    test_num = int(tot_len * 0.1)
    if( isTrain ): 
        index = 1 + np.random.choice(tot_len - test_num, batch_size, False)
    else:
        index = 1 + tot_len - test_num + np.random.choice(test_num, batch_size, False)
    
    images = np.array([image_load_helpers.get_image(image_paths[i], 128).reshape([64 * 64 * 3]) for i in index]) / 255.
    labels = label_data[index-1]
    
    return images, labels

def shuffle(images, targets):
    rng_state = np.random.get_state()
    np.random.shuffle(images)
    np.random.set_state(rng_state)
    np.random.shuffle(targets)

def cifar10_load():
    path = './pretrained_models/cifar10/data/cifar10_train/cifar-10-batches-py/'
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    data = []
    targets = []
    for batch in batches:
        with open(path + batch, 'rb') as file_handle:
            batch_data = pickle.load(file_handle)
            batch_data['data'] = (batch_data['data'] / 255.0)
            data.append(batch_data['data'])
            targets.append(batch_data['labels'])
    with open(path + 'test_batch') as file_handle:
        batch_data = pickle.load(file_handle)
        batch_data['data'] = (batch_data['data'] / 255.0)
        return np.vstack(data), np.concatenate(targets), batch_data['data'], batch_data['labels']

    
def load_fmnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784) / 255.0

    return images, labels

def F_MNIST_load():
    tr_image, tr_label = load_fmnist('./data/f_mnist/', 'train')
    ts_image, ts_label = load_fmnist('./data/f_mnist/', 't10k')
    
    #shuffle(tr_image, tr_label)
    #shuffle(ts_image, ts_label)

    return (tr_image, tr_label, ts_image, ts_label)


def MNIST_load():
    filepath = './data/mnist/mnist_py3k.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz'

    if not os.path.isfile(filepath):
        print ("Couldn't find MNIST dataset in ./data, downloading...")
        urllib.urlretrieve(url, filepath)
    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    tr_image, tr_label = train_data
    ts_image, ts_label = test_data
    shuffle(tr_image, tr_label)
    shuffle(ts_image, ts_label)

    return (tr_image, tr_label, ts_image, ts_label)


def adv_load_dataset(batch_size, data):
    train_data, train_target, test_data, test_target = data

    def train_epoch():
        tot_len = train_data.shape[0]
        i = np.random.randint(0, batch_size)
        while (i + batch_size < tot_len):
            yield (np.copy(train_data[i:i + batch_size, :]), np.copy(train_target[i:i + batch_size]))
            i = i + batch_size

    return train_epoch


def mix_image(image, random_map):
    flat = image.flatten()
    new_flat = np.ndarray(flat.shape)
    for index, rand_index in enumerate(random_map):
        new_flat[index] = flat[rand_index]

    return new_flat.reshape(image.shape)


def unmix_image(image, random_map):
    flat = image.flatten()
    new_flat = np.ndarray(flat.shape)
    for index, rand_index in enumerate(random_map):
        new_flat[rand_index] = flat[index]

    return new_flat.reshape(image.shape)

def save_zip_data(object, filename, bin = 1):
    """Saves a compressed object to disk"""
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, bin))
    file.close()

def create_MNIST_mixed():
    # Load MNIST data
    filepath = './data/mnist/mnist_py3k.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist_py3k.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in ./data, downloading...")
        urllib.urlretrieve(url, filepath)

    with gzip.open(filepath, 'rb') as f:
        data_chunks = pickle.load(f)

    # Create Random mapping of pixels
    image_size = 784
    indices = [index for index in range(0, image_size)]
    np.random.shuffle(indices)
    rand_map = indices

    # Mix MNIST data
    mixed_data = []
    for i, chunk in enumerate(data_chunks):
        images = chunk[0]
        labels = chunk[1]

        mixed_images = []
        for image in images:
            mixed_image = mix_image(image, rand_map)
            mixed_images.append(mixed_image)
        mixed_images = np.asarray(mixed_images)

        mixed_chunk = (mixed_images, labels)
        mixed_data.append(mixed_chunk)

    # Save mixed data
    filepath = './data/mixed_mnist_py3k.pkl.gz'
    save_zip_data(mixed_data, filepath)

    # Save random map
    filepath = './data/random_map.pkl'
    pickle.dump(rand_map, open(filepath, "wb"))

def MNIST_load_mixed():
    filepath = './data/mixed_mnist_py3k.pkl.gz'

    if not os.path.isfile(filepath):
        print('Mixed MNIST dataset not found at:')
        print(filepath)
        print('Creating mixed MNIST dataset...')
        create_MNIST_mixed()

    with gzip.open(filepath, 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f)

    tr_image, tr_label = train_data
    ts_image, ts_label = test_data
    shuffle(tr_image, tr_label)
    shuffle(ts_image, ts_label)

    return (tr_image, tr_label, ts_image, ts_label)

def file_exists(path):
    return os.path.isfile(path)

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    nh, nw = rows, int(n_samples/rows) + 1

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))
    
    if X.ndim == 4:
        # BCHW -> BHWC
        if( X.shape[1] == 3 ):
            X = X.transpose(0,2,3,1)
            
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)

def test_2d():
    it, TRAIN_SIZE, TEST_SIZE = 0, 260520, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        x0, y0 = 4 * (np.random.randint(3, size=2) - 1)
        r = np.random.normal(0, 0.5)
        t = np.random.uniform(0, 6.3)
        xy = np.matrix([x0 + (r**2)*math.cos(t), y0 + (r**2)*math.sin(t)])
        #x0, y0 = np.random.uniform(0, 1, size=2)
        #xy = np.matrix([x0 + 1, y0 + 1])
        label = 1

        it = it + 1
        if( it < TRAIN_SIZE ):
            train_data.append(xy)
            train_target.append(label)
        else:
            test_data.append(xy)
            test_target.append(label)

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plt.scatter(np.asarray(train_data[:, 0]).flatten(), np.asarray(train_data[:, 1]).flatten(), s=0.4, c='b', alpha=0.7)

    fig.savefig('train.png')
    plt.close()

    return train_data, train_target, test_data, test_target

# Toy Testset
def swiss_load():
    it, TRAIN_SIZE, TEST_SIZE = 0, 65536, 2000
    train_data, test_data = [], []
    train_target, test_target = [], []
    while( it < TRAIN_SIZE + TEST_SIZE ):
        t = np.random.uniform(0, 10)

        xy = 0.5*np.matrix([t*math.cos(2*t), t*math.sin(2*t)])
        label = int(t < 5)

        it = it + 1
        if( it < TRAIN_SIZE ):
            train_data.append(xy)
            train_target.append(label)
        else:
            test_data.append(xy)
            test_target.append(label)

    train_data = np.vstack(train_data)
    test_data = np.vstack(test_data)

    return train_data, train_target, test_data, test_target

def dynamic_load_dataset(batch_size, load_func, *func_args):
    label_data, image_paths = load_func(*func_args)
    tot_len = len(label_data)
    
    test_len = int(tot_len*0.1)
    train_len = tot_len - test_len
    
    def train_epoch():
        i = np.random.randint(0, batch_size)
        while(i + batch_size < train_len):
            data, label = load_func(label_data, image_paths, batch_size, isTrain=True)
            yield data, label
            i = i + batch_size
            
    def test_epoch():
        i = 0
        while(i + batch_size < test_len):
            data, label = load_func(label_data, image_paths, batch_size, isTrain=False)
            yield data, label
            i = i + batch_size
            
    return train_epoch, None, test_epoch
    
def load_dataset(batch_size, load_func, dynamic_load = False, *func_args):
    if( dynamic_load ):
        return dynamic_load_dataset(batch_size, load_func, *func_args)
    data = load_func(*func_args)
    train_data, train_target, test_data, test_target = data
    test_size = batch_size

    def train_epoch():
        tot_len = train_data.shape[0]
        i = np.random.randint(0, batch_size)
        while(i + batch_size < tot_len):
            yield (np.copy(train_data[i:i+batch_size, :]), np.copy(train_target[i:i+batch_size]))
            i = i + batch_size

    def test_epoch():
        tot_len = test_data.shape[0]
        i = 0
        #i = np.random.randint(0, test_size)
        while(i + test_size < tot_len):
            yield (np.copy(test_data[i:i+test_size, :]), np.copy(test_target[i:i+test_size]))
            i = i + batch_size

    return train_epoch, data, test_epoch

def batch_gen(gens, use_one_hot_encoding=False, out_dim=-1, num_iter=-1):
    it = 0
    while (it < num_iter) or (num_iter < 0):
        it = it + 1

        for images, targets in gens():
            if( use_one_hot_encoding ):
                n = len(targets)
                one_hot_code = np.zeros((n, out_dim))
                one_hot_code[range(n), targets] = 1
                yield images, one_hot_code
            else:
                yield images, targets

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def get_factors(number):
    upper_num = int(math.ceil(number*0.5))
    factors = [x for x in range(1, upper_num+1) if number % x == 0]
    factors.append(number)
    return factors

def subplot_values(num_figures):
    factors = get_factors(num_figures)
    sqroot = math.sqrt(num_figures)
    factor1 = min(factors, key=lambda x: abs(x - sqroot))
    factor2 = int(num_figures/factor1)

    return factor1, factor2

def make_one_hot(coll):
    onehot = np.zeros((coll.shape[0], coll.max() + 1))
    onehot[np.arange(coll.shape[0]), coll] = 1
    return onehot