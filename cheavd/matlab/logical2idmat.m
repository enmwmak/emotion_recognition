function idmat = logical2idmat(spk_logical)
% Convert from spk_logical format to speaker id format (matrix W in Campbell's paper)

[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
numSpks = length(unique(spk_ids));
numVecs = length(spk_logical);
idmat = zeros(numVecs,numVecs);
for i = 1:numSpks,
    spk_sessions = find(spk_ids == i);  % Sessions indexes of speaker i
    idmat(spk_sessions,spk_sessions) = 1;
end
return;