% Prepart Leave-One-Speaker-Out (LOSO) cross-validation set so that the
% reported results are speaker-independent
% Input:
%   datadir       - Folder containing the data file
%   datafile      - Name of data file containing the full dataset
%   nFeatures     - No. of feature per class; Selected features are the union of all classes
%   n_ev          - No. of NAP directions to be projected out.
% Output:
%   Write CV dataset to datadir/emodb_trn_loso*.mat and datadir/emodb_tst_loso*.mat
% Example:
%   prepare_loso_cv('../data/IS09_emotion');
%   prepare_loso_cv('../data/IS09_emotion', 'emodb_full.mat');
%   prepare_loso_cv('../data/IS09_emotion', 'emodb_exBoredom.mat');
%   prepare_loso_cv('../data/IS11_speaker_state', 'emodb_exBoredom.mat', []);
%   prepare_loso_cv('../data/IS11_speaker_state', 'emodb_exDisgust.mat', 500);
%   prepare_loso_cv('../data/IS11_speaker_state', 'emodb_full.mat', [], 1);
function prepare_loso_cv(datadir, datafile, nFeatures, n_ev)
addpath '~/so/Matlab/nap';

if ~exist('datafile','var')
    datafile = 'emodb_full.mat';
end

if ~exist('nFeatures','var')
    nFeatures = inf;
end

if ~exist('n_ev','var')
    n_ev = 0;
end

% Load data to obtain a structure containing all info of emodb
matfile = strcat(datadir,'/',datafile);
fprintf('Loading %s\n', matfile);
data = load(matfile);

% Select features. If no. of features is inf, no need to select.
if nFeatures ~= inf,
    fmask = select_fdr_features(data.x, data.y, nFeatures);
    ufmask = zeros(1, size(fmask,2));
    for i=1:size(fmask,1),
        ufmask = double(or(ufmask, fmask(i,:)));
    end
    data.x(:,ufmask==0) = [];
end
    
% Create leave-one-speaker-out cross-validation set
P = eye(size(data.x,2));
nSpks = length(unique(data.spknums));
for i = 1:nSpks,
    trnidx = find(data.spknums ~= i);
    trn = extract_data(data, trnidx);
    if n_ev > 0,
        [trn.x, P] = make_nap_data(trn.x, trn.targets, n_ev);
    end
    loso_cv_file1 = sprintf('%s/emodb_trn_loso%d.mat',datadir,i);        
    save_data_task1(loso_cv_file1, trn);
    loso_cv_file2 = sprintf('%s/gender_trn_loso%d.mat',datadir,i);
    save_data_task2(loso_cv_file2, trn);
    
    tstidx = find(data.spknums == i);
    tst = extract_data(data, tstidx);
    tst.x = (P*tst.x')';
    loso_cv_file1 = sprintf('%s/emodb_tst_loso%d.mat',datadir,i);
    save_data_task1(loso_cv_file1, tst);
    loso_cv_file2 = sprintf('%s/gender_tst_loso%d.mat',datadir,i);
    save_data_task2(loso_cv_file2, tst);    
end

% Export total number of selected features so that it can be retrieved by bash shell
fprintf('nFeatures=%d\n', size(data.x,2));

%% 
% Private function
function dataout = extract_data(datain, idx)
dataout.fvectors = datain.fvectors(idx,:);
dataout.genders = datain.genders(idx);
dataout.nclasses = datain.nclasses;
dataout.spknums = datain.spknums;
dataout.targets = datain.targets(idx,:);
dataout.x = datain.x(idx,:);
dataout.y = datain.y(idx);

function save_data_task1(filename, data)
x = data.x;
y = data.y;
fprintf('Writing %s\n', filename);
save(filename, 'x', 'y');

function save_data_task2(filename, data)
x = data.x;
y = zeros(length(data.y),1);
for i=1:length(y),
    if data.genders(i) == 'f',
        y(i) = 1;
    end
end
fprintf('Writing %s\n', filename);
save(filename, 'x', 'y');

