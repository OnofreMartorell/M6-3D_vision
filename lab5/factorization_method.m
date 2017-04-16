function [  Pproj,  Xproj ] = factorization_method( x1, x2 , init)
% ToDo: create the function 'factorization_method' that computes a
% projective reconstruction with the factorization method of Sturm and
% Triggs '1996
% This function returns an estimate of:
%       Pproj: 3*Npoints x Ncam matrix containing the camera matrices
%       Xproj: 4 x Npoints matrix of homogeneous coordinates of 3D points
%
% As a convergence criterion you may compute the Euclidean
% distance (d) between data points and projected points in both images
% and stop when (abs(d - d_old)/d) < 0.1 where d_old is the distance
% in the previous iteration.
%x1, x2: points in homogeneous coordinates

[~, n] = size(x1);

if strcmp(init, 'ones')
    Lambda = ones(2, n);
else
    x_cat = {x1, x2};
    % j point i view
    lamda1j = 1;
    for i = 1:length(x_cat)
        F = fundamental_matrix(x_cat{i}, x_cat{1});
        e = null(F);
        for j = 1:n 
            num = x_cat{1}(:, j)'*F*cross(e, x_cat{i}(:,j));
            den = norm(cross(e,x_cat{i}(:,j))).^2;
            Lambda(i,j) = (num/den).*lamda1j;
        end
    end
    Lambda(1,:)=1;
end

threshold = 0.1;

[x1_norm, H1] = normalise2dpts(x1);
[x2_norm, H2] = normalise2dpts(x2);


x_matrix = cat(1, x1_norm, x2_norm);
finished = false;
d_old = Inf;
while not(finished)
    %normalice Lambda
    for i = 1:2
        %Normalize rows
        norm_row = sqrt(sum(Lambda.^2, 2));
        for r = 1:2
           Lambda(r, :) =  Lambda(r, :)/norm_row(r);
        end    
        %Normalize cols
        norm_col = sqrt(sum(Lambda.^2, 1));
        for c = 1:n
           Lambda(:, c) =  Lambda(:, c)/norm_col(c);
        end
    end
    Lambda_M = cat(1, Lambda(1,:), Lambda(1,:), Lambda(1,:), Lambda(2,:), Lambda(2,:), Lambda(2,:));
    M = Lambda_M.*x_matrix;
    [U, D, V] = svd(M, 'econ');
    D_4 = D(:, 1:4);
    
    V_4 = V(:, 1:4);
    P_M = U*D_4;
    X_M = V_4';
    
    P_1 = P_M(1:3, :);
    P_2 = P_M(4:6, :);
    x1_reproj = P_1*X_M;
    x2_reproj = P_2*X_M;
    
    dist_im1 = sum(sum((x1_reproj - x1_norm).^2));
    dist_im2 = sum(sum((x2_reproj - x2_norm).^2));
    d = dist_im1 + dist_im2;
    finished = (abs(d - d_old)/d) < threshold;

    d_old = d;
    Lambda(1,:) = x1_reproj(3,:);
    Lambda(2,:) = x2_reproj(3,:);
end

P_1_end = pinv(H1)*P_1;
P_2_end = pinv(H2)*P_2;

Pproj = cat(1, P_1_end, P_2_end);
Xproj = X_M;
end

