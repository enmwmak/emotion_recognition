% Remove speaker with small number of utts
function [w, spk_logical] = remove_bad_spks(w, spk_logical, min_num_utts)
[~, ~, spk_ids]=unique(spk_logical);    % spk_ids contains indexes to unique speakers
numSpks = length(unique(spk_ids));
rm_idx = [];
for i=1:numSpks,
    idx = find(spk_ids == i);
    if (length(idx) < min_num_utts),
        rm_idx = [rm_idx; idx];
    end
end
spk_logical(rm_idx) = [];
w(rm_idx,:) = [];