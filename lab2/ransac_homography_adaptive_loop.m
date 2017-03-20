function [H, idx_inliers] = ransac_homography_adaptive_loop(x1, x2, th, max_it)

[~, Npoints] = size(x1);

% ransac
it = 0;
best_inliers = [];
while it < max_it

    % Take four pairs of points from the pairs of matches given
    points = randomsample(Npoints, 4);
    % Compute the homography using this four points
    if all(~diff(x1(:, points)'))
        continue;
        %disp('The matches fall into the same point.');
    elseif all(~diff(x2(:, points)'))
        %disp('The matches fall into the same point.');
        continue;
    end
    %disp('gets in');
    H = homography2d(x1(:, points), x2(:, points)); % ToDo: you have to create this function
    inliers = compute_inliers(H, x1, x2, th);
    
    % test if it is the best model so far
    if length(inliers) > length(best_inliers)
        best_inliers = inliers;
    end
    
    % update estimate of max_it (the number of trials) to ensure we pick,
    % with probability p, an initial data set with no outliers
    fracinliers = length(inliers)/Npoints;
    pNoOutliers = 1 -  fracinliers^4;
    pNoOutliers = max(eps, pNoOutliers);  % avoid division by -Inf
    pNoOutliers = min(1 - eps, pNoOutliers);% avoid division by 0
    p = 0.99;
    max_it = log(1 - p)/log(pNoOutliers);
    
    it = it + 1;
end

% compute H from all the inliers
H = homography2d(x1(:,best_inliers), x2(:,best_inliers));
idx_inliers = best_inliers;
end

function idx_inliers = compute_inliers(H, x1, x2, th)
% Check that H is invertible
if abs(log(cond(H))) > 15
    idx_inliers = [];
    return
end

% compute the symmetric geometric error

 d2 = sum((euclid(x1) - euclid(inv(H)*x2)).^2,1) + sum((euclid(x2) - euclid(H*x1)).^2,1);
 idx_inliers = find(d2 < th.^2);
end

%
% function diff =compute_symmetric_difference(x1,x2,H)
% 
% diff = norm(euclid(x1)-euclid(inv(H)*x2)) + norm(euclid(x2) - euclid(H*x1));
% 
% end


function xn = normalise(x)
xn = x ./ repmat(x(end,:), size(x,1), 1);
end

function item = randomsample(npts, n)
a = 1:npts;
item = zeros(1,n);
for i = 1:n
    % Generate random value in the appropriate range
    r = ceil((npts - i + 1).*rand);
    item(i) = a(r);       % Select the rth element from the list
    a(r)    = a(end - i + 1); % Overwrite selected element
end                       % ... and repeat
end