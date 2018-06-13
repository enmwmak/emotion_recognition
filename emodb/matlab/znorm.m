function z = znorm(x, mu, sigma)

% Perform z-norm
z = zeros(size(x));
for i = 1:size(x,1),
    z(i,:) = (x(i,:) - mu)./sigma;
end

% Whitening + lennorm
%z = whiten(z);
%z = len_norm(z);