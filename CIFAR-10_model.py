import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import pickle
import urllib.request

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import SGD

def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    d = pickle.load(f, encoding="bytes")
    for k, v in d.items():
        del(d[k])
        d[k.decode("utf8")] = v
    f.close()
    data = d["data"]
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    final /= 255
    final -= .5
    labels2 = np.zeros((len(labels), 10))
    labels2[np.arange(len(labels2)), labels] = 1

    return final, labels

def load_batch(fpath):
    f = open(fpath,"rb").read()
    size = 32*32*3+1
    labels = []
    images = []
    for i in range(10000):
        arr = np.fromstring(f[i*size:(i+1)*size],dtype=np.uint8)
        lab = np.identity(10)[arr[0]]
        img = arr[1:].reshape((3,32,32)).transpose((1,2,0))

        labels.append(lab)
        images.append((img/255)-.5)
    return np.array(images),np.array(labels)


class CIFAR:
    def __init__(self):
        train_data = []
        train_labels = []

        if not os.path.exists("cifar-10-batches-bin"):
            urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                                       "cifar-data.tar.gz")
            os.popen("tar -xzf cifar-data.tar.gz").read()


        for i in range(5):
            r,s = load_batch("cifar-10-batches-bin/data_batch_"+str(i+1)+".bin")
            train_data.extend(r)
            train_labels.extend(s)

        train_data = np.array(train_data,dtype=np.float32)
        train_labels = np.array(train_labels)

        self.test_data, self.test_labels = load_batch("cifar-10-batches-bin/test_batch.bin")


        VALIDATION_SIZE = 5000

        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]
def train(data, file_name, params, num_epochs=64, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()
    model.add(Conv2D (params[0], (3, 3),activation='relu', kernel_initializer='he_uniform', padding='same',input_shape= data.train_data.shape[1:]))

    model.add(BatchNormalization())

    model.add(Conv2D(params[1], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(params[2], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(params[3], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.3))

    model.add(Conv2D(params[4], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())

    model.add(Conv2D(params[5], (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

    model.add(BatchNormalization())



    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(params[6], activation='relu', kernel_initializer='he_uniform'))


    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(10),activation='softmax')



    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=sgd,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)


    if file_name != None:
        model.save(file_name)

    return model



if not os.path.isdir('/model'):
    os.makedirs('/model')

train(CIFAR(), "model/cifar",  [32,32,64,64,128,128,128])
