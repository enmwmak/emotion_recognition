function acc = get_spkid_acc(scoremat, spk_logical)
% Return speaker ID accuracy based on a pairwise score matrix and spkID info

[~,~,spkid] = unique(spk_logical);
n_spks = max(spkid);
n_ivecs = length(spk_logical);

% Find the session index of each test speaker
sessions = cell(n_spks,1);
for s = unique(spkid)',
   sessions{s} = find(spkid == s); 
end

% For each test i-vectors, find the average scores of individual speakers
% For each speaker, we consider all of his/her i-vectors (excluding the one
% that is used as the test i-ivector) as enrollment i-vectors
scores = zeros(n_ivecs, n_spks);
maxpos = zeros(n_ivecs, 1);
for i = 1:n_ivecs,
    for s = 1:n_spks,
        sess = setdiff(sessions{s},i);                  % Do not count self-comparison
        scores(i,s) = mean(scoremat(i,sess));
    end
    [~,maxpos(i)] = max(scores(i,:));
end

% For each test i-vectors, if the maxpos matches the spkid, the i-vec is correctly classified;
n_correct = 0;
for i = 1:n_ivecs,
    if maxpos(i) == spkid(i),
        n_correct = n_correct + 1;
    end
end
acc = n_correct/n_ivecs;