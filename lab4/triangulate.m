function [X] = triangulate(x1, x2, P1, P2, imsize)

% The entries are (x1, x2, P1, P2, imsize), where:
%           - x1, and x2 are the Euclidean coordinates of two matching 
%             points in two different images.
%           - P1 and P2 are the two camera matrices
%           - imsize is a two-dimensional vector with the image size

H = [2/imsize(1) 0 -1;
    0 2/imsize(2) -1;
    0 0 1];

x1_norm = euclid(H*homog(x1));
x2_norm = euclid(H*homog(x2));

P1_norm = H*P1;
P2_norm = H*P2;


A = [x1_norm(1)*P1_norm(3, :) - P1_norm(1, :);
    x1_norm(2)*P1_norm(3, :) - P1_norm(2, :);
    
    x2_norm(1)*P2_norm(3, :) - P2_norm(1, :);
    x2_norm(2)*P2_norm(3, :) - P2_norm(2, :)];

[~, ~, V] = svd(A);

X = V(:, end);
end