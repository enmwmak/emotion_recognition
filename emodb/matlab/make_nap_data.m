% Create a dataset by performing NAP on the original dataset
% The NAP parameters are estimated from the training dataset
function [Xnap, P] = make_nap_data(X, targets, n_ev)

% Apply NAP projection. Make the indexes to emotion classes consecutive for nap_train.m
nClasses = size(targets,2);
[~,trainLabels] = max(targets,[],2);
trnVecs = []; trnLbls = [];
for k=1:nClasses,
    idx = find(trainLabels == k);
    trnVecs = [trnVecs; X(idx,:)];
    trnLbls = [trnLbls; trainLabels(idx)];
end
W = logical2idmat(trnLbls);
[P,V,lambda] = nap_train(trnVecs', W, n_ev);

% Project emotion vectors
Xnap = (P*X')';
