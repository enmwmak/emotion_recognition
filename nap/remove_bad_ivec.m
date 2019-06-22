% Remove i-vecs with big norm
function [w, spk_logical] = remove_bad_ivec(w, spk_logical, normlimit)
N = length(spk_logical);
normw = zeros(N,1);
for i=1:size(w,1), 
    normw(i) = norm(w(i,:)); 
end
idx = find(normw < normlimit);
w = w(idx,:);
spk_logical = spk_logical(idx);
return;