% To use the functions in this library, you need to load the weka.jar into memory
% To do that, you may use addjavapath as follows
% Execute this script from anywhere
%   run('~/so/Matlab/arff/setup_arff.m');

% To read .arff file and convert to Matlab data, do the following:
% wekaOBJ = loadARFF('test.arff');
% [mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(wekaOBJ);
% data = mdata(:,2:end-1);      % data is an N x nFeatures matrix

% M.W. Mak, May 2016

arffpath = '~/so/Matlab/arff';

% Add arff to the Matlab path
addpath(arffpath);

cdir = pwd;
cd(arffpath);
javaaddpath('weka.jar');
cd(cdir);
