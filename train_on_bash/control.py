import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
import numpy as np
import scipy.optimize as opt
import sys, os, random, gzip
import tensorflow as tf
from keras import backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import pandas as pd

from sklearn.metrics import roc_curve, auc
import csv
import pickle
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#         print(e)

from tensorflow import keras
# from my_particlenet import get_MCGCNN, get_particle_net
# from MCGCNN2 import get_MCGCNN, get_particle_net
from MCGCNN3 import get_MCGCNN, get_particle_net
# from tf_keras_model import get_particle_net, get_particle_net_lite

##---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

##self

# model_type = 'particle_net'
input_shapes={'points': (100, 2), 'features': (100, 5), 'mask': (100, 1)}
num_classes = 2
w =  -2 ## weights you can choose (it is added by myself, and I use the 5-th as the weight term)
model = get_MCGCNN(num_classes, input_shapes, w=w, K=7)
epochs = 200
batch_size = 200
# model = get_particle_net(num_classes, input_shapes, w=w)
# Prepare model model saving directory.


model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
# optimizer=keras.optimizers.Adam(learning_rate=lr_schedule(0))
model.summary()

import os
model_type = "MCGCNN"
save_dir = './model_checkpoints'
model_name = '%s_model.test.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

progress_bar = keras.callbacks.ProgbarLogger()
csv_logger = keras.callbacks.CSVLogger('./myparticlenet_training_log.csv')
earlystop = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=1e-4,
                            patience=10,
                            verbose=1,
                            mode='min', baseline=None, ## 'min' 
                            restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001)
callbacks = [checkpoint, csv_logger, progress_bar, earlystop ]




X_val = np.load("/home/ja2006203966/script/Network/practice/X_val.npy",allow_pickle=True)
X_val = X_val.item()
y_val = np.load("/home/ja2006203966/script/Network/practice/y_val.npy",allow_pickle=True)
# n=1000000
valm = [99999,0]
earlystop_n = 2

N=10 # num you want to split
j = 0
i = 0


for i in range(N):
    print("epochs= " ,j, "/",epochs, "\tbatch= ", i, "/",N)
    X_train = np.load("/home/ja2006203966/script/Network/practice/X_train"+str(i)+".npy",allow_pickle=True)
    X_train = X_train.item()
    y_train = np.load("/home/ja2006203966/script/Network/practice/y_train"+str(i)+".npy",allow_pickle=True)
    model.fit(X_train, y_train,batch_size=batch_size,epochs=1, validation_data=(X_val, y_val), shuffle=True , callbacks=callbacks)
    val = model.evaluate(X_val,  y_val, verbose=2)
    LOSS1 = pd.read_csv('./myparticlenet_training_log.csv')
    LOSS1['epoch'][0] = j
    if (i==0)&(j==0):
        LOSS0 = LOSS1
        LOSS0.to_csv("./checkpoint.csv")
    else:
        LOSS0 = LOSS0.append(LOSS1, ignore_index=True)
        LOSS0.to_csv("./checkpoint.csv")
    if (round(val[0],4)<round(valm[0],4)):
        model.save('./MCGCNN_best.h5')

        valm[0] = val[0]
        valm[1] = j

j = j+1          
np.save("./j",j)
np.save("./valm",valm)
model.save('./MCGCNN.h5')
os.system("nohup python3 -u ./train.py > ./train_"+str(j)+".log 2>&1 &")


        
