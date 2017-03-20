function Error_x = gs_errfunction( P0, Xobs ) 
% ToDo: create this function that we need to pass to the lsqnonlin function
% NOTE: gs_errfunction should return E(X) and not the sum-of-squares E=sum(E(X).^2)) that we want to minimize. 
% (E(X) is summed and squared implicitly in the lsqnonlin algorithm.) 

% Xobs = [ x(:) ; xp(:) ];     % The column vector of observed values (x and x')
% P0 = [ Hab(:) ; x(:) ];      % The parameters or independent variables
length_x_hat = (length(P0) - 9)/2;

Homography = reshape( P0(1:9), 3, 3 );
xx = mat2cell(Xobs, [ length(Xobs)/2 length(Xobs)/2]);
x_v = reshape(xx{1}', 2, length(xx{1})/2);
xp_v = reshape(xx{2}', 2, length(xx{2})/2);
x = homog(x_v);
xp = homog(xp_v);

x_hat = homog(reshape(P0(10:length(P0)), 2, length_x_hat));
x_hat_p = Homography*x_hat;

dist_left = sqrt(sum((euclid(x) - euclid(x_hat)).^2, 1));
dist_right = sqrt(sum((euclid(xp) - euclid(x_hat_p)).^2, 1));

Error_x = [dist_left; dist_right];
end