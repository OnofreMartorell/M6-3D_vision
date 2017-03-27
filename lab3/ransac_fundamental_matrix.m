function [F, idx_inliers] = ransac_fundamental_matrix(p1, p2, th)

[~, num_pairs] = size(p1);

% ransac
it = 0;
max_it = Inf;
best_inliers = [];
% probability that at least one random sample set is free of outliers
p = 0.999; 
while it < max_it
    
    points = randomsample(num_pairs, 8);
    F = fundamental_matrix(p1(:, points), p2(:, points));

    inliers = compute_inliers(F, p1, p2, th);
    
    % test if it is the best model so far
    if length(inliers) > length(best_inliers)
        best_inliers = inliers;
    end    
    
    % update estimate of max_it (the number of trials) to ensure we pick, 
    % with probability p, an initial data set with no outliers
    fracinliers =  length(inliers)/num_pairs;
    pNoOutliers = 1 -  fracinliers^4;
    pNoOutliers = max(eps, pNoOutliers);  % avoid division by -Inf
    pNoOutliers = min(1 - eps, pNoOutliers);% avoid division by 0
    max_it = log(1 - p)/log(pNoOutliers);
    
    it = it + 1;
end

% compute F from the best_inliers
F = fundamental_matrix(p1(:, best_inliers), p2(:, best_inliers));
idx_inliers = best_inliers;
end

function idx_inliers = compute_inliers(F, p1, p2, th)
[~, num_pairs] = size(p1);
F_p1 = F*p1;
F_p2 = F'*p2;
p2_F_p1 = zeros(1, num_pairs);
for i = 1:num_pairs
    p2_F_p1(i) = p2(:, i)'*F_p1(:, i);
end    

F_p1 = F_p1.^2;
F_p2 = F_p2.^2;
% Compute the Sampson error
d2 = (p2_F_p1.^2)./(F_p1(1, :) + F_p1(2, :) + F_p2(1, :) + F_p2(2, :));
idx_inliers = find(d2 < th.^2);
end

function item = randomsample(npts, n)
a = 1:npts;
item = zeros(1, n);
for i = 1:n
    % Generate random value in the appropriate range
    r = ceil((npts - i + 1).*rand);
    item(i) = a(r);       % Select the rth element from the list
    a(r)    = a(end - i + 1); % Overwrite selected element
end
end