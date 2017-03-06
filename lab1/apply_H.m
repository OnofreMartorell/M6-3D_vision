%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%    Apply Homgraphy     %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [ destination_Image ] = apply_H( original_Image , H )
%apply_H Performs a specific homography to a given image
%   Inpunt:
%       -Original Image: Image to be transformed
%       -H: matrix 3x3 which defines the homography
%    
%   Output:
%       -Destination Image: Image after homography
%       -First the destination image size is computed from the four
%       external points transformed.
%       -Then, a transformation is made so the points do not fall outside
%       the border.
%       -Afterwards, the points are draw on its place.
%
    close all
    

    siz = size(original_Image);
    
    transposed_Im = zeros(siz(2), siz(1), 3);
    for i = 1:3
        transposed_Im(:, :, i) = original_Image(:, :, i)';
    end
    original_Image = transposed_Im;
%     siz = size(original_Image);
%   The homography is called T because maybe further changes will be done
    T0 = (H);
    
%   The corners are localized and transformed so the destination size 
%   can be computed    
    
    p1 = double([1, 1, 1]');
    p2 = double([1, siz(2), 1]');
    p3 = double([siz(1), 1, 1]');
    p4 = double([siz(1), siz(2), 1]');
    
    % Transformation of cornersin homogeneous coordinates
    p_transf_homogeneous = [(T0*p1)'; (T0*p2)'; (T0*p3)'; (T0*p4)'];
    
    ph1 = T0*p1;
    ph2 = T0*p2;
    ph3 = T0*p3;
    ph4 = T0*p4;
    
    % Cartesian coordinates of transformed corners
    p_transf_cartesian = [p_transf_homogeneous(:, 1)./p_transf_homogeneous(:, 3) ...
                            p_transf_homogeneous(:, 2)./p_transf_homogeneous(:, 3)];
    ph1c = [ph1(1)/ph1(3),  ph1(2)/ph1(3)];
    ph2c = [ph2(1)/ph2(3),  ph2(2)/ph2(3)];
    ph3c = [ph3(1)/ph3(3),  ph3(2)/ph3(3)]; 
    ph4c = [ph4(1)/ph4(3),  ph4(2)/ph4(3)]; 

    % Upleft and downright corners
    [~, i ] = min(p_transf_cartesian(:, 1));
    newzero = min([ph1c; ph2c; ph3c; ph4c]);
    newend = max([ph1c; ph2c; ph3c; ph4c]);
    
    ph1c1 = ph1c - newzero;
    ph2c1 = ph2c - newzero;
    ph3c1 = ph3c - newzero; 
    ph4c1 = ph4c - newzero; 
    
    % Iniziatization of the destination image
    destination_Image = uint8(zeros(round(newend(1) - newzero(1)), round(newend(2) - newzero(2)), 3));
    size_destination = size(destination_Image);
    
    T = inv(T0);
        
    
%   Every pixel is backtransformed
for row = 1:size_destination(1)
    for col = 1:size_destination(2)
        p_transf_cart = [row col] + newzero;
%       The pixel is transformed to homogeneous coordinates
        p = double([p_transf_cart 1]');
        ph = T*p;
           
%       cartesian coordinates are obtained
        pcart(1) = ph(1)/ph(3);
        pcart(2) = ph(2)/ph(3);
        cim = pcart(1);
        rim = pcart(2);
        
%       Rounding so an integer is computed since they are coordinates        
        cc = round(cim);
        rr = round(rim);

%       The pixels are placed on the destination image
%       (First is check the pixel belongs to the original image)
        if  rr > 0 && rr <= size(original_Image, 1) && cc > 0 && cc <= size(original_Image, 2)
            destination_Image(row, col, :) = original_Image(rr, cc, :);
%             Vq = interp2(X,Y,V,Xq,Yq) returns interpolated values of a 
% function of two variables at specific query points using linear interpolation. 
% The results always pass through the original sampling of the function. X and Y 
% contain the coordinates of the sample points. 
% V contains the corresponding function values at each sample point. 
% Xq and Yq contain the coordinates of the query points.
        end
    end
end
% I2 = destination_Image;
% figure; imshow(uint8(I2));
% hold on
% scatter(ph1c1(1), ph1c1(2) + 1, 'filled', 'g')
% hold on
% scatter(ph2c1(1) + 1, ph2c1(2), 'filled', 'y')
% hold on
% scatter(ph3c1(1), ph3c1(2), 'filled', 'r')
% hold on
% scatter(ph4c1(1), ph4c1(2), 'filled', 'b')
end