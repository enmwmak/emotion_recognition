% Return a nClasses x dim matrix. For each row, entries with '1' represent selected features.

function [fmask,scores] = select_fdr_features(fvectors, labels, nFeatures)

dim = size(fvectors, 2);                % Feature vectors in rows
nClasses = max(labels)+1;               % Class labels start from 0
fmask = zeros(nClasses, dim);
scores = zeros(nClasses, dim);
for i=1:nClasses,
    posIdx = labels==i-1;
    negIdx = labels~=i-1;
    [fmask(i,:), scores(i,:)] = fdr(fvectors(posIdx,:), fvectors(negIdx,:), nFeatures);
end

%% Private function
function [mask,scores] = fdr(Xpos, Xneg, nFeatures)
dim = size(Xpos,2);
mask = zeros(1, dim);
mu_p = mean(Xpos,1);
sigma_p = std(Xpos,[],1);
mu_n = mean(Xneg,1);
sigma_n = std(Xneg,[],1);
fdr = (((mu_p-mu_n)).^2)./(sigma_p.^2+sigma_n.^2);
[~,idx] = sort(fdr,'descend');
idx(nFeatures+1:end) = [];
mask(idx) = 1;
scores = fdr;

