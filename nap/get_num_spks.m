function n_spks = get_num_spks(spk_logical)
% Return the number of speakers in the dataset

[~,~,spkid] = unique(spk_logical);
n_spks = max(spkid);
