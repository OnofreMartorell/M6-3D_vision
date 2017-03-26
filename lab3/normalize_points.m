function [x_norm, T] = normalize_points(x)

[~, length_x] = size(x);
x_euclid = euclid(x);

centroid = mean(x_euclid(1:2, :), 2);

x_centered = x_euclid - repmat(centroid, 1, length_x);
average_dist = mean(sqrt(sum(x_centered.^2, 1)));
scale = sqrt(2)/average_dist;

T = [scale 0 -scale*centroid(1)
    0 scale -scale*centroid(2)
    0 0 1];

x_norm = T*x;

end