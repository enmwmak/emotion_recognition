% Import all .arff files under the arff/ folder and save the data to
% an N x D matrix, where N is the no. of vectors and D is the feature 
% dimension
% To init ARFF Matlab library in ~/so/Matlab/arff (your system may be different)
% run('~/so/Matlab/arff/setup_arff.m');

clear; close all;

% Define constants and options
featureType = 'IS11_speaker_state';  % Feature type extracted by using different opensmile config.
%featureType = 'IS09_emotion';  
%exclude = 'boredom';                 % Make results comparable with "Speaker-sensitive emotion recognition via ranking: 
                                     % Studies on acted and spontaneous speech"
%exclude = 'disgust';                 % Disgust was excluded in "Psychological Motivated Multi-Stage Emotion Classification Exploiting Voice Quality Features"
exclude = 'none';

% Init ARFF library
if (exist('weka2matlab.m','file') == 0),
    run('~/so/Matlab/arff/setup_arff.m');
end

% Define ARFF dir and output matfile
arffdir = sprintf('../arff/%s',featureType);
if strcmp(exclude,'boredom'),
    matfile = sprintf('../data/%s/emodb_exBoredom.mat', featureType);
elseif strcmp(exclude,'disgust'),
    matfile = sprintf('../data/%s/emodb_exDisgust.mat', featureType);
else
    matfile = sprintf('../data/%s/emodb_full.mat',featureType);
end
    
% Define the hashmap for mapping emotion classes to emotion class labels
if strcmp(exclude,'boredom'),
    labels = {'W','E','A','F','T','N'};
    classes = {1,2,3,4,5,6};    
elseif strcmp(exclude,'disgust'),
    labels = {'W','L','A','F','T','N'};
    classes = {1,2,3,4,5,6};    
else    
    labels = {'W','L','E','A','F','T','N'};
    classes = {1,2,3,4,5,6,7};
end    
mapObj = containers.Map(labels,classes);

% Define the hashmap for mapping speaker number (1-10) to speaker IDs
spkmap = containers.Map({'03','08','09','10','11','12','13','14','15','16'},...
                        { 1,   2,   3,   4,   5,   6,   7,   8,   9,   10});

% Define the hashmap for mapping speaker ID to gender
gendermap = containers.Map({'03','08','09','10','11','12','13','14','15','16'},...
                           { 'm', 'f', 'f', 'm', 'm', 'm', 'f', 'f', 'm', 'f'});
                    
% Get the file info of all files under the arff folder
files = dir(arffdir);

% Read all files in arffdir
fvectors = []; emotionID = []; y = []; spknums=[]; genders=[];
for i=3:length(files),
    emoID = files(i).name(6);
    if isKey(mapObj,emoID),
        emotionID = [emotionID; mapObj(emoID)];
        spknums = [spknums spkmap(files(i).name(1:2))];
        genders = [genders gendermap(files(i).name(1:2))];
        arfffile = strcat(arffdir,'/',files(i).name);
        fprintf('Reading %s\n', arfffile);
        wekaOBJ = loadARFF(arfffile);
        [mdata,featureNames,targetNDX,stringVals,relationName] = weka2matlab(wekaOBJ);
        data = mdata(:,2:end-1);      % data is a 1 x nFeatures matrix
        fvectors = [fvectors; data];
    end
end

% Remove features with very small variances
rmidx = std(fvectors,1,1)<5e-6;
fvectors(:,rmidx) = [];

% Find stats and max and min of features from training data
[mu, sigma] = get_feature_stats(fvectors);
xmax = max(fvectors,[],1);
xmin = min(fvectors,[],1);

% Apply z-norm to fvectors and save structure's fields to file
fvectors = znorm(fvectors, mu, sigma);

% Convert emotionID to 1-of-K format
targets = zeros(size(fvectors,1),length(labels));
for i = 1:size(fvectors,1),
    targets(i,emotionID(i)) = 1;
end

% Save data matrix and target to .mat file
x = fvectors;
y = emotionID-1;
nclasses = length(labels);
fprintf('Saving %s\n', matfile);
save(matfile, 'x', 'y', 'fvectors', 'targets', 'nclasses', 'spknums', 'genders');

% Load CHEAVD raw data (all of them) and use selected features found from EmoDB
% cheavd = load('../../cheavd/data/IS11_speaker_state/cheavd_raw.mat');
% x = cheavd.x;
% x(:,rmidx) = [];
% [mu, sigma] = get_feature_stats(x);
% x = znorm(x, mu, sigma);
% y = cheavd.y;
% matfile = '../../cheavd/data/IS11_speaker_state/cheavd_nom.mat';
% fprintf('Saving %s\n', matfile);
% save(matfile, 'x', 'y');


