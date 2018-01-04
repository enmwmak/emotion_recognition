# To run this script using Python3.6, assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 dnn_emotion_cls.py
#   source deactivate tf-py3.6
# The following configuration achieves 83.93%
#       optimizer = 'rmsprop'
#       activation = 'sigmoid'
#       n_features = 1781
#       fs_method = 'none'        (This means FDR done in Matlab is the best)

# To run this script using Python2.7, assuming that Anaconda2 environment "tf-py2.7"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda2/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py2.7
#   python dnn_emotion_cls.py
#   source deactivate tf-py2.7

# Run ../matlab/prepare_loso_cv.m in Matlab to create the LOSO CV training and testing files in .mat
# before running this program

# Author: M.W. Mak, Dept. of EIE, HKPolyU
# Last update: 27 Oct. 2017

from __future__ import print_function
import numpy as np
import scipy.io as sio
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import regularizers
import FeatureSelection as fs
from keras import backend as K
from keras.callbacks import EarlyStopping


def main():
    # Define constant
    optimizer = 'adam'    # Can be 'adam', 'sgd', or 'rmsprop'
    activation = 'relu'   # Can be 'sigmoid', 'tanh', 'softplus', 'softsign', 'relu'
    datadir = 'data/IS11_speaker_state'           # 'data/IS09_emotion' or 'data/IS11_speaker_state'
    # datadir = 'data/IS09_emotion'           # 'data/IS09_emotion' or 'data/IS11_speaker_state'
    n_hiddens = [100, 100]
    n_correct = 0
    n_samples = 0
    n_features = 2500                       # Totally 382 features for IS09_emotion; 4354 for IS11_speaker_state
    fs_method = 'none'                      # 'univ', 'rfe', 'lasso', 'l1norm' or 'none'
    C = 0.01
    rfe_step = 1
    n_speakers = 10                         # There are 10 speakers in EmoDB

    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    print('Start cross validation...')
    for i in range(1, n_speakers + 1):

        print('Speaker ' + str(i) + ': ', end='')
        trn_file = datadir + '/' + 'emodb_trn_loso' + str(i) + '.mat'
        tst_file = datadir + '/' + 'emodb_tst_loso' + str(i) + '.mat'

        # Load trn and tst data
        trn_data = sio.loadmat(trn_file)
        tst_data = sio.loadmat(tst_file)

        # Retreive training and test data
        x_train = np.array(trn_data['x'], dtype='float32')
        y_train = np.array(trn_data['y'].ravel(), dtype='int32')
        y_train_ohe = np_utils.to_categorical(y_train)
        x_test = np.array(tst_data['x'], dtype='float32')
        y_test = np.array(tst_data['y'].ravel(), dtype='int32')

        if fs_method == 'rfe':
            # Select features by RFE-SVM, remove 10% of feature for each iteration (step=0.1)
            x_train, x_test = fs.rfe_select_features(x_train, y_train, x_test, n_fs=n_features,
                                                  penalty=C, step=rfe_step)
        elif fs_method == 'l1norm':
            # Select features by L1-norm
            x_train, x_test = fs.l1norm_select_features(x_train, y_train, x_test, penalty=C)
        elif fs_method == 'lasso':
            # Select features by LASSO
            x_train, x_test = fs.lasso_select_features(x_train, y_train, x_test, alpha=0.001)
        elif fs_method == 'univ':
            x_train, x_test = fs.univ_select_features(x_train, y_train, x_test, n_fs=n_features)
        elif fs_method == 'fdr':
            x_train, x_test = fs.fdr_select_features(x_train, y_train, x_test, n_fs=n_features)
        elif fs_method == 'none':
            pass
        print('No. of selected features = %d, ' % x_train.shape[1], end='')

        # Train DNN
        model = train_dnn(x_train, y_train_ohe, n_hiddens, optimizer, activation,
                          n_epochs=20, bat_size=50)

        # Test DNN
        train_acc, dummy, dummy = test_dnn(x_train, y_train, model)
        print('DNN Training accuracy: %.2f%% ' % (train_acc * 100), end='')

        # Get DNN accuracy of current speaker
        test_acc, n_c, n_s = test_dnn(x_test, y_test, model)
        n_correct = n_correct + n_c
        n_samples = n_samples + n_s
        print('DNN Test accuracy: %.2f%% ' % (test_acc * 100))

        # Release GPU memory for next fold
        # K.clear_session()

    n_classes = y_train_ohe.shape[1]
    print('Overall DNN accuracy for %d classes with %d features: %.2f%%' %
          (n_classes, n_features, (float(n_correct)/float(n_samples) * 100)))


def get_accuracy(pred_labels, true_labels):
    y_pred = np.empty((0, ), dtype='int')
    y_true = np.empty((0, ), dtype='int')
    for i in range(len(pred_labels)):
        y_pred = np.hstack((y_pred, pred_labels[i]))
        y_true = np.hstack((y_true, true_labels[i]))
    n_correct = np.sum(y_true == y_pred, axis=0)
    acc = 100 * n_correct / y_true.shape[0]
    return acc


def train_dnn(x_train, y_train_ohe, n_hiddens, optimizer, act, n_epochs=20, bat_size=50):
    np.random.seed(1)

    # Create a DNN
    model = Sequential()

    # Define number of hidden layers and number of nodes in each layer according to n_hiddens
    model.add(Dense(n_hiddens[0], input_dim=x_train.shape[1], activation=act, name='Layer-1'))
    # model.add(BatchNormalization(name='L1-BN1'))
    model.add(Dropout(0.2, name='L1-Dropout1'))
    for i in range(1, len(n_hiddens)):
        model.add(Dense(n_hiddens[i], name='Layer-%d' % (i+1), kernel_regularizer=regularizers.l2(0.1)))
        # model.add(BatchNormalization(name='L%d-BN%d' % (i+1, i+1)))
        model.add(Activation(act, name='L%d-Act%d' % (i+1, i+1)))
        model.add(Dropout(0.2, name='L%d-Dropout%d' % (i+1, i+1)))
    model.add(Dense(y_train_ohe.shape[1], name='Layer-BeforeSM', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Activation('softmax', name='Layer-AfterSM'))

    # Define loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    # Define the callback for early stoping
    early_stop = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=0, verbose=1, mode='max')

    # Perform training
    model.fit(x_train, y_train_ohe, epochs=n_epochs, batch_size=bat_size, verbose=0,
              validation_data=(x_train, y_train_ohe), shuffle=True)

    return model


def test_dnn(X, y, model):
    y_pred = model.predict_classes(X, verbose=0)
    n_correct = np.sum(y == y_pred, axis=0)
    n_samples = X.shape[0]
    test_acc = n_correct / float(n_samples)
    return test_acc, n_correct, n_samples


main()

