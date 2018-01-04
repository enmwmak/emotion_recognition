% Read the training, validation, and test data (in .mat) files.
% Extract the relevant features by FDR. Save the data with selected
% features to .mat files.
% Input:
%   dataset        - Name of dataset (e.g., 'cheavd', 'emodb')
%   datadir        - Directory containing the data in .mat files
%   nFeatures      - No. of features per class (the total no. of
%                    selected features for all classes could be larger)
%   n_ev           - No. of NAP directions to be projected out
% Output:          - .mat files containing x and y with selected features
% Example:
%   prepare_data('cheavd', '../data/IS09_emotion');
%   prepare_data('cheavd', '../data/IS11_speaker_state');
%   prepare_data('cheavd', '../data/IS11_speaker_state', [], 1);
%   prepare_data('cheavd', '../data/IS11_speaker_state', 1500);
function prepare_data(dataset, datadir, nFeatPerClass, n_ev)

addpath 'nap';

if ~exist('nFeatPerClass','var')
    nFeatPerClass = inf;
end

if ~exist('n_ev','var')
    n_ev = 0;
end

% Inut files
trnfile = sprintf('%s/%s_trn.mat', datadir, dataset);
valfile = sprintf('%s/%s_val.mat', datadir, dataset);
tstfile = sprintf('%s/%s_tst.mat', datadir, dataset);
allfile = sprintf('%s/%s_all.mat', datadir, dataset);

% Load data to get x(1:nData,1:dim) and y(1:nData)
fprintf('Loading %s\n', trnfile); trn = load(trnfile);
fprintf('Loading %s\n', valfile); val = load(valfile);
fprintf('Loading %s\n', tstfile); tst = load(tstfile);
fprintf('Loading %s\n', allfile); all = load(allfile);

% Perform FDR feature selection
if nFeatPerClass ~= inf,
    fmask = select_fdr_features(trn.x, trn.y, nFeatPerClass);
    ufmask = zeros(1, size(fmask,2));
    for i=1:size(fmask,1),
        ufmask = double(or(ufmask, fmask(i,:)));
    end
    fidx = find(ufmask==1);
    trn.x = trn.x(:,fidx);
    val.x = val.x(:,fidx);
    tst.x = tst.x(:,fidx);
    all.x = all.x(:,fidx);
end

% Perform NAP projection
if n_ev > 0,
    [trn.x, P] = make_nap_data(trn.x, trn.targets, n_ev);
    val.x = (P*val.x')';
    tst.x = (P*tst.x')';
end

% Output files
trnfile = sprintf('%s/%s_trn_fs.mat', datadir, dataset);
valfile = sprintf('%s/%s_val_fs.mat', datadir, dataset);
tstfile = sprintf('%s/%s_tst_fs.mat', datadir, dataset);
allfile = sprintf('%s/%s_all_fs.mat', datadir, dataset);

% Save data with selected features
fprintf('Saving x and y with %d selected features to .mat files\n', size(trn.x,2));
fprintf('Saving %s\n', trnfile); save(trnfile, '-struct', 'trn');
fprintf('Saving %s\n', valfile); save(valfile, '-struct', 'val');
fprintf('Saving %s\n', tstfile); save(tstfile, '-struct', 'tst');
fprintf('Saving %s\n', allfile); save(allfile, '-struct', 'all');

% Export total number of selected features so that it can be retrieved by bash shell
fprintf('nFeatures=%d\n', size(trn.x,2));

