function Error_x = gs_errfunction( P0, Xobs ) 
% ToDo: create this function that we need to pass to the lsqnonlin function
% NOTE: gs_errfunction should return E(X) and not the sum-of-squares E=sum(E(X).^2)) that we want to minimize. 
% (E(X) is summed and squared implicitly in the lsqnonlin algorithm.) 

% Xobs = [ x(:) ; xp(:) ];     % The column vector of observed values (x and x')
% P0 = [ Hab(:) ; x(:) ];      % The parameters or independent variables
length_x_hat = (length(P0) - 9)/2;

Homography = reshape( P0(1:9), 3, 3 );
xx = reshape(Xobs, 2, length(Xobs)/2);
x = [xx(:, 1:length(xx)/2); ones(1, length_x_hat)];
xp = [xx(:, length(xx)/2 + 1:length(xx)); ones(1, length_x_hat)];

x_hat = [reshape(P0(10:length(P0)), 2, length_x_hat); ones(1, length_x_hat)];
x_hat_p = Homography*x_hat;

dist_left = norm(x - x_hat);
dist_right = norm(xp - x_hat_p);

Error_x = [dist_left dist_right];
end