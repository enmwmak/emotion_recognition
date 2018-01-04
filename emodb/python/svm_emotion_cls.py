# To run this script using Python3.6, assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 svm_emotion_cls.py
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
#   python svm_emotion_cls.py
#   source deactivate tf-py2.7

# Run ../matlab/prepare_loso_cv.m in Matlab to create the LOSO CV training and testing files in .mat
# before running this program

# Author: M.W. Mak, Dept. of EIE, HKPolyU
# Last update: 27 Oct. 2017

from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn.svm import SVC
import FeatureSelection as fs


def main():
    # Define constant
    datadir = 'data/IS11_speaker_state'           # 'data/IS09_emotion' or 'data/IS11_speaker_state'
    # datadir = 'data/IS09_emotion'           # 'data/IS09_emotion' or 'data/IS11_speaker_state'
    n_features = 500                       # Totally 382 features for IS09_emotion; 4354 for IS11_speaker_state
    fs_method = 'none'                       # 'univ', 'rfe', 'lasso', 'l1norm', 'fdr' or 'none'
    C = 0.01                                # Penalty factor for RFE-SVM feature selection
    rfe_step = 1                            # Step for RFE-SVM feature selection
    n_speakers = 10                         # There are 10 speakers in EmoDB

    svm_pred = []
    svm_true = []
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
            # Select features by FDR
            x_train, x_test = fs.fdr_select_features(x_train, y_train, x_test, n_fs=n_features)
        elif fs_method == 'none':
            n_features = x_train.shape[1]
        print('No. of selected features = %d, ' % x_train.shape[1], end='')

        # Train an SVM classifier
        svc = SVC(C=10.0, gamma='auto', kernel='rbf')
        svc.fit(x_train, y_train)

        # Test the SVM classifier
        pred = svc.predict(x_test)
        print('SVM accuracy = %.2f%%' % get_accuracy([pred], [y_test]))
        svm_pred.append(svc.predict(x_test))
        svm_true.append(y_test)

    n_classes = np.max(y_train) + 1
    print('Overall SVM accuracy for %d classes with %d features/class: %.2f%%' %
          (n_classes, n_features, get_accuracy(svm_pred, svm_true)))


def get_accuracy(pred_labels, true_labels):
    y_pred = np.empty((0, ), dtype='int')
    y_true = np.empty((0, ), dtype='int')
    for i in range(len(pred_labels)):
        y_pred = np.hstack((y_pred, pred_labels[i]))
        y_true = np.hstack((y_true, true_labels[i]))
    n_correct = np.sum(y_true == y_pred, axis=0)
    acc = 100 * n_correct / y_true.shape[0]
    return acc


main()

