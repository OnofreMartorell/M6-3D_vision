function f = fundamental_matrix(pts1h, pts2h)
% % Normalize the points
% num = cast(size(pts1h, 2), integerClass);
 [pts1h, t1] = normalise2dpts(pts1h);
 [pts2h, t2] = normalise2dpts(pts2h);
% 
% % Compute the constraint matrix
% m = coder.nullcopy(zeros(num, 9, outputClass));
for idx = 1: size(pts1h, 2)
  w(idx,:) = [...
    pts1h(1,idx)*pts2h(1,idx), pts1h(2,idx)*pts2h(1,idx), pts2h(1,idx), ...
    pts1h(1,idx)*pts2h(2,idx), pts1h(2,idx)*pts2h(2,idx), pts2h(2,idx), ...
                 pts1h(1,idx),              pts1h(2,idx), 1];
end

% Find out the eigen-vector corresponding to the smallest eigen-value.
[~, ~, vm] = svd(w);
f = reshape(vm(:, end), 3, 3)';

% Enforce rank-2 constraint
[u, s, v] = svd(f);
s(end) = 0;
f = u * s * v';

% Transform the fundamental matrix back to its original scale.
f = t2' * f * t1;

% Normalize the fundamental matrix.
f = f / norm(f);
if f(end) < 0
  f = -f;
end
