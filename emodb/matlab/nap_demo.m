% File: nap_demo.m
% Demonstration of NAP functions using i-vectors

clear; close all;
num_proj = 100;

% Load dataset
dataset = load('mat/male_target-tel-06dB_mix_t500_w_1024c.mat');

% Define training (5717) and test data (1439)
n_data = length(dataset.spk_logical);
n_trn = 5717;
n_tst = n_data - n_trn;
trnidx = 1:n_trn;
tstidx = n_trn+1:n_data;
trn.w = dataset.w(trnidx,:);                    % w(:,500) contains 500-dim i-vecs in rows
trn.spk_logical = dataset.spk_logical(trnidx);  % spk_logical(1:n_data) contains spkID in text form
tst.w = dataset.w(tstidx,:);
tst.spk_logical = dataset.spk_logical(tstidx);

% Remove i-vecs with big norm
[trn.w, trn.spk_logical] = remove_bad_ivec(trn.w, trn.spk_logical, 30); 
[tst.w, tst.spk_logical] = remove_bad_ivec(tst.w, tst.spk_logical, 30); 

% Remove speakers with less than 5 utts
[trn.w, trn.spk_logical] = remove_bad_spks(trn.w, trn.spk_logical, 5);
[tst.w, tst.spk_logical] = remove_bad_spks(tst.w, tst.spk_logical, 5);

% Print number of training and test speakers and i-vectors
fprintf('No. of training speakers = %d\n', get_num_spks(trn.spk_logical));
fprintf('No. of training i-vectors = %d\n', length(trn.spk_logical));
fprintf('No. of test speakers = %d\n', get_num_spks(tst.spk_logical));
fprintf('No. of test i-vectors = %d\n', length(tst.spk_logical));

% Compute pairwise cosine-distance score matrix of test i-vectors (before NAP)
scoremat = pairwise_cds(tst.spk_logical, tst.w);
save 'mat/scoremat.mat' scoremat;
%load('mat/scoremat.mat');

% Compute speaker id accuracy of test i-vectors (before NAP)
acc = get_spkid_acc(scoremat, tst.spk_logical);
fprintf('\nSpeaker id acc before NAP = %.2f%%\n',acc*100);

% Convert trn.spk_logical to spkid matrix (W in Campbell's paper)
W = logical2idmat(trn.spk_logical);

% Compute NAP projection matrix
[P,V,lambda] = nap_train(trn.w', W, num_proj);

% Project test i-vectors
tst_w_nap = P*tst.w';

% Compute pairwise cosine-distance score matrix of test i-vectors (after NAP)
scoremat_nap = pairwise_cds(tst.spk_logical, tst_w_nap');
save 'mat/scoremat_nap.mat' scoremat_nap;

acc = get_spkid_acc(scoremat_nap, tst.spk_logical);
fprintf('\nSpeaker id acc after NAP = %.2f%%\n',acc*100);



