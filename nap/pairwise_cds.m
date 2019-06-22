function scoremat = pairwise_cds(spk_logical, w)
% Note: w contains row vectors

n_tst = length(spk_logical);
scoremat = zeros(n_tst,n_tst);
norm_w = zeros(n_tst,1);

% Compute the norm of all vectors for speeding up
for i=1:n_tst,
    norm_w(i) = norm(w(i,:));
end

for i=1:n_tst,
    %fprintf('Scoring utt %d of %d\r',i,n_tst);
    for j=i:n_tst,
        scoremat(i,j) = w(i,:)*w(j,:)'/(norm_w(i)*norm_w(j));
        scoremat(j,i) = scoremat(i,j);
    end
end