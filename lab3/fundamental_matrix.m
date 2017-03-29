function F = fundamental_matrix(pts1h, pts2h)

% Normalizing the points
[pts1hn, T1] = normalise2dpts(pts1h);
[pts2hn, T2] = normalise2dpts(pts2h);

% Creating matrix W
[~, num_pairs] = size(pts1hn);
W = zeros(num_pairs, 9);

for i = 1: size(pts1hn, 2)
    W(i,:) = [...
        pts1hn(1,i)*pts2hn(1,i), pts1hn(2,i)*pts2hn(1,i), pts2hn(1,i), ...
        pts1hn(1,i)*pts2hn(2,i), pts1hn(2,i)*pts2hn(2,i), pts2hn(2,i), ...
        pts1hn(1,i),              pts1hn(2,i), 1];
end

% Obtaining V matrix so last column is used as solution for F.
[~, ~, v2] = svd(W);
F = reshape(v2(:, end), 3, 3)';

% Rank2 constraint
[u, s, v] = svd(F);
s(end) = 0;
F = u * s * v';

% Transforming the fundamental matrix back.
F = T2' * F * T1;
F = F / norm(F);
if F(end) < 0
    F = -F;
end
end