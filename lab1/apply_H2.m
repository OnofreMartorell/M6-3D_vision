% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %%    Apply Homgraphy    %%
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
function [ destination_Image ] = apply_H2(original_Image , H)
% %apply_H Performs a specific homography to a given image
% %   Inpunt:
% %       -Original Image: Image to be transformed
% %       -H: matrix 3x3 which defines the homography
% %
% %   Output:
% %       -Destination Image: Image after homography
% %       -First the destination image size is computed from the four
% %       external points transformed.
% %       -Then, a transformation is made so the points do not fall outside
% %       the border.
% %       -Afterwards, the points are draw on its place.

size_original









end
% size_original = size(original_Image);
% 
% transposed_Im = permute(original_Image, [2 1 3]);
% size_transposed = size(transposed_Im);
% 
% %   The corners are localized and transformed so the destination size
% %   can be computed.
% 
% p1 = double([1, 1, 1]');
% p2 = double([1, size_original(2), 1]');
% p3 = double([size_original(1), 1, 1]');
% p4 = double([size_original(1), size_original(2), 1]');
% 
% % Transformation of corners in homogeneous coordinates
% p_transf_homogeneous = [(H*p1)'; (H*p2)'; (H*p3)'; (H*p4)'];
% 
% % Cartesian coordinates of transformed corners
% p_transf_cartesian = [p_transf_homogeneous(:, 1)./p_transf_homogeneous(:, 3) ...
%     p_transf_homogeneous(:, 2)./p_transf_homogeneous(:, 3)];
% 
% % Upleft and downright corners
% newzero = min(p_transf_cartesian);
% newend = max(p_transf_cartesian);
% 
% % Initialization of the destination image
% destination_Image = uint8(zeros(round(newend(1) - newzero(1)), round(newend(2) - newzero(2)), 3));
% size_destination = size(destination_Image);
% 
% % [X, Y] = meshgrid(1:size_transposed(1), 1:size_transposed(2));
% %   Every pixel is backtransformed
% 
% % x = reshape(X, size_transposed(1)*size_transposed(2), 1);
% % y = reshape(Y, size_transposed(1)*size_transposed(2), 1);
% % im1 = reshape(double(transposed_Im(:, :, 1)), size_transposed(1)*size_transposed(2), 1);
% % im2 = reshape(double(transposed_Im(:, :, 2)), size_transposed(1)*size_transposed(2), 1);
% % im3 = reshape(double(transposed_Im(:, :, 3)), size_transposed(1)*size_transposed(2), 1);
% % % F1 = scatteredInterpolant(x, y, im1);
% % % F2 = scatteredInterpolant(x, y, im2);
% % % F3 = scatteredInterpolant(x, y, im3);
% % 
% % for row = 1:size_destination(1)
% %     for col = 1:size_destination(2)
% %         p_transf_cart = [row col] + newzero;
% %         %       The pixel is transformed to homogeneous coordinates
% %         p = double([p_transf_cart 1]');
% %         ph = H\p;
% % 
% %         %       Cartesian coordinates are obtained
% %         pcart(1) = ph(1)/ph(3);
% %         pcart(2) = ph(2)/ph(3);
% %         cim = pcart(1);
% %         rim = pcart(2);
% % 
% %         %       Rounding so an integer is computed since they are coordinates
% %         cc = round(cim);
% %         rr = round(rim);
% % 
% %         %       The pixels are placed on the destination image
% %         %       (First is check the pixel belongs to the original image)
% %         if  rr > 0 && rr <= size_transposed(1) && cc > 0 && cc <= size_transposed(2)
% %             interp2
% %             destination_Image(row, col, 1) = F1(cim, rim);
% %             destination_Image(row, col, 2) = F2(cim, rim);
% %             destination_Image(row, col, 3) = F3(cim, rim);
% %         end
% %     end
% % end
% % 
% 
% 
% 
% %Points of the original image
% % [X, Y] = meshgrid(1:size_original(1), 1:size_original(2));
% % Create a meshgrid of all the points we need to find its value
% [X, Y, Z] = meshgrid(1 + newzero(1):size_destination(1) + newzero(1), 1 + newzero(2):size_destination(2) + newzero(2), 1);
% 
% % Points in homogeneous coordinates. Each column is a point
% points_dest_hom = cat(3, X, Y, Z);
% points_dest_hom = reshape(points_dest_hom, 1, size_destination(1)*size_destination(2), 3);
% points_dest_hom = permute(points_dest_hom, [3 2 1]);
% 
% % Points in cartesian coordinates.
% [X, Y] = meshgrid(1:size_destination(1), 1:size_destination(2));
% points_dest_cart = cat(3, X, Y);
% points_dest_cart = reshape(points_dest_cart, 1, size_destination(1)*size_destination(2), 2);
% points_dest_cart = permute(points_dest_cart, [3 2 1]);
% % points_destination = points_dest_hom(1:2, :);
% 
% % Transformation of all the point by homography
% points_original_hom = inv(H)*points_dest_hom;
% 
% % Points in cartesian coordinates
% points_original_cart = points_original_hom(1:2, :);
% points_original_cart(1, :) = points_original_hom(1, :)./points_original_hom(3, :);
% points_original_cart(2, :) = points_original_hom(2, :)./points_original_hom(3, :);
% 
% % Points that lie inside the area of the original image. Vector of logical
% original_valid_logical = round(points_original_cart);
% original_valid_logical(1, :) = original_valid_logical(1, :) > 0 & original_valid_logical(1, :) <= size_transposed(1);
% original_valid_logical(2, :) = original_valid_logical(2, :) > 0 & original_valid_logical(2, :) <= size_transposed(1);
% original_valid_logical = original_valid_logical(1, :) & original_valid_logical(2, :);
% 
% % Points that lie inside the area of the original image. Vector of
% % coordinate points
% points_original_valid = points_original_cart(:, original_valid_logical);
% points_destination_valid = round(points_dest_cart(:, original_valid_logical));
% 
% [X, Y] = meshgrid(1:size_transposed(1), 1:size_transposed(2));
% [Xq, Yq] = meshgrid(points_original_valid(1, :), points_original_valid(2, :));
% % Vq = interp2(X, Y, transposed_Im, Xq, Yq);
% 
% for k = 1:3
%     Vq = interp2(X, Y,...
%         transposed_Im(:, :, k), Xq, Yq);       
%     destination_Image(points_destination_valid, k) = Vq(new_points);
% end
% 
% %         [Xq,Yq] = meshgrid(-3:0.25:3)
% %         Vq = interp2(X,Y,V,Xq,Yq) returns interpolated values of a function
% % of two variables at specific query points using linear interpolation.
% % The results always pass through the original sampling of the function.
% % X and Y contain the coordinates of the sample points.
% % V contains the corresponding function values at each sample point.
% % Xq and Yq contain the coordinates of the query points.
% 
% % values_f1 = F1(points_original_valid');
% % values_f2 = F2(points_original_valid');
% % values_f3 = F3(points_original_valid');
% % for k = 1:length(valuesf1)
% %     destination_Image(points_destination_valid(1, k), points_destination_valid(2, k), :) = ...
% %         [values_f1(k) values_f2(k) values_f3(k) ];
% % end    
% % destination_Image(points_destination_valid, 2)
% end