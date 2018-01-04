function [P,V,lambda] = nap_train(A, W, num_proj)
% Implementation of Campbell's NAP
% W.M. Campbell et al. "SVM Based Speaker Verification Using a GMM
% Supervector Kernel and NAP Variability Compensation", ICASSP'06.
% Inputs:
%   A          - Matrix A in Eq. 12. Each column contains a vector
%   W          - Matrix W in Eq. 12. W(i,j)=1 if Column i and Column j are from the same speaker
%                The indexes to the same speaker must be consecutive, e.g., i=1,...,n(1)
%                correspond to Spk1, i=n(1)+1:n(1)+1:n(2) correspond to Spk2
%   num_proj   - No. of dimensions to be projected out
% Output:
%   P          - Projection matrix (I-VV')
%   V          - Eigenvectors corresponding to nuisance directions
%   lambda     - Eigenvalues corresponding to nuisance directions
% Example usage: 
%   See nap_demo.m
% Author:
%   M.W. Mak, The HKPolyU

% No. of utterances
num_utts = size(W, 1);

% Feature dimension
fdim = size(A,1);

% Prepare matrix J in Campbell's paper
i = 1; j = 1;
while i<=num_utts,
    n(j) = length(find(W(:,i)==1));     % n(j) contains the no. of utterances from speaker j
    i = i + n(j);
    j = j + 1;
end
J = zeros(num_utts,num_utts);
J(1:n(1),1:n(1)) = sqrt(n(1))*(eye(n(1))-(1/n(1))*ones(n(1),1)*ones(n(1),1)');
k = n(1);
for j = 2:length(n),
    J(k+1:k+n(j),k+1:k+n(j)) = sqrt(n(j))*(eye(n(j))-(1/n(j))*ones(n(j),1)*ones(n(j),1)');
    k = k + n(j);
end

% Solve the eigen problem, i.e. finding v in Eq. 13
% [eigV, eigD] = eig(A*(diag(W*ones(num_utts,1))-W)*A');
n_ev = min([num_proj*2 fdim-2]);            % Compute double the number of necessary eigenvectors
if (size(A,1) > size(A,2)),
    % Feature dim > No. of vectors
    % Instead of solving eig(AJA'), we solve eig(B'B), where B=AJ.  
    % See http://en.wikipedia.org/wiki/Eigenface
    B = A*J;
    if n_ev > size(B,1),
        n_ev = size(B,1);
    end
    [eigU, eigD] = eigs(double(B'*B), n_ev);
    eigV = B*eigU;
else
    % As the no. of vectors is larger than feature dim, we perform eigen analysis directly
    [eigV, eigD] = eigs(A*J*A',n_ev);
end   
for i = 1:n_ev,
    eigV(:,i) = eigV(:,i)/norm(eigV(:,i));      % Make ||v_i|| = 1
end

% Select the components with real and positive eigenvalues
eigvalues = diag(eigD);
for i=1:length(eigvalues),
    if isreal(eigvalues(i)) == 0;
        eigvalues(i) = 0;
    end
end
idx = find(eigvalues>0);
V = eigV(:,idx);
lambda = eigvalues(idx);

% Compute projection matrix P
fdim = size(V,1);
sum_v = zeros(fdim,fdim);
for i = 1:num_proj,
    sum_v = sum_v + V(:,i)*V(:,i)';
end
P = eye(fdim) - sum_v;

