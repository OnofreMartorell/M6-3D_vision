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
	Lambda = 0;
end

threshold = 0.1;

[x1_norm, H1] = normalise2dpoints(x1);
[x2_norm, H2] = normalise2dpoints(x2);


x_matrix = cat(1, x1_norm, x2_norm);
finished = 0;
d_old = Inf;
while finished
	%normalice Lambda
	for i = 1:2
		%Normalize rows
		norm_row = sum(Lambda, 1);
		
		%Normalize cols
		norm_col = sum(Lambda, 2);
	end
	Lambda_M = cat(1, Lambda(1,:), Lambda(1,:), Lambda(1,:), Lambda(2,:), Lambda(2,:), Lambda(2,:));
	M = Lambda_M*x_matrix;
	[U, D, V] = svd(M);
	D_4 = D(:, 1:4);
	D_other = D(:, 5:end);
	
	P_M = U*D_4;
	X_M = D_other*V';
	
	P_1 = P_M(1:3, :);
	P_2 = P_M(4:6, :);
	x1_reproj = P_1*X_M;
	x2_reproj = P_2*X_M;
	
	dist_im1 = sum(sum((x1_reproj - x1_norm).^));
	dist_im2 = sum(sum((x2_reproj - x2_norm).^));
	d = dist_im1 + dist_im2;
	finished = abs(d - d_old)/d) < threshold;
	d_old = d;
	Lambda(1,:) = x1_reproj(3,:);
	Lambda(2,:) = x2_reproj(3,:);
end

P_1_end = pinv(H1)*P_1;
P_2_end = pinv(H2)*P_2;

Pproj = cat(1, P_1_end, P_2_end);
Xproj = X_M;
end

