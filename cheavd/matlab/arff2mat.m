% Import all .arff files under the arff/ folder and save the data to
% an N x D matrix, where N is the no. of vectors and D is the feature 
% dimension
% To init ARFF Matlab library in ~/so/Matlab/arff (your system may be different)
% run('~/so/Matlab/arff/setup_arff.m');

% Note: This script require weka2matlab. 
% You may download the library from https://www.mathworks.com/matlabcentral/fileexchange/21204-matlab-weka-interface?focused=5103451&tab=function
% You may change the script setup_arff.m to suit your need.

clear; close all;

if (exist('weka2matlab.m','file') == 0),
    run('~/so/Matlab/arff/setup_arff.m');
end

% feaType = 'IS09_emotion';
feaType = 'IS11_speaker_state';

% Output .mat files containing 'n_emos','x', and 'y' 
trn_matfile = sprintf('../data/%s/cheavd_trn.mat', feaType);
val_matfile = sprintf('../data/%s/cheavd_val.mat', feaType);
tst_matfile = sprintf('../data/%s/cheavd_tst.mat', feaType);
raw_matfile = sprintf('../data/%s/cheavd_raw.mat', feaType);

% Label files
trn_labfile = '/corpus/cheavd/data/train/train_label.txt';
val_labfile = '/corpus/cheavd/data/val/val_label.txt';
tst_labfile = '/corpus/cheavd/data/test/test_label.txt';

% Arff files
trn_arffdir = sprintf('../arff/%s/train',feaType);
val_arffdir = sprintf('../arff/%s/val',feaType);
tst_arffdir = sprintf('../arff/%s/test',feaType);

% Define label to classID map
labels = {'angry','disgust','happy','neutral','sad','surprise'};
classes = {1,2,3,4,5,6};
labmap = containers.Map(labels,classes);
n_emos = length(classes);

% Read training data and save them in .mat file
trn = get_cheavd_data(trn_arffdir, trn_labfile, labmap);
val = get_cheavd_data(val_arffdir, val_labfile, labmap); 
tst = get_cheavd_data(tst_arffdir, tst_labfile, labmap); 

% Save unprocessed (raw) data to .mat file
x = [trn.x; val.x; tst.x];
y = [trn.y; val.y; tst.y];
fprintf('Saving raw data to %s\n', raw_matfile);
save(raw_matfile, 'x', 'y');

% Remove features with 0 variances
%X = [trn.x; val.x];
X = trn.x;
rmidx = std(X,1,1)==0;
X(:,rmidx) = [];

% Find stats and max and min of features from training data
[mu, sigma] = get_feature_stats(X);
xmax = max(X,[],1);
xmin = min(X,[],1);

% Apply z-norm to input vectors and save structure's fields to file
trn.x(:,rmidx) = [];
trn.x = znorm(trn.x, mu, sigma);
save(trn_matfile, '-struct', 'trn');

val.x(:,rmidx) = [];
val.x = znorm(val.x, mu, sigma);
save(val_matfile, '-struct', 'val');

tst.x(:,rmidx) = [];
tst.x = znorm(tst.x, mu, sigma);
save(tst_matfile, '-struct', 'tst');



break;

