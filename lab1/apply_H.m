%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                      %%    Apply Homgraphy    %%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
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
    
    size_original = size(original_Image);
    
    transposed_Im = permute(original_Image, [2 1 3]);
    size_transposed = size(transposed_Im);
    
%   The corners are localized and transformed so the destination size 
%   can be computed.
    
    p1 = double([1, 1, 1]');
    p2 = double([1, size_original(2), 1]');
    p3 = double([size_original(1), 1, 1]');
    p4 = double([size_original(1), size_original(2), 1]');
    
    % Transformation of corners in homogeneous coordinates
    p_transf_homogeneous = [(H*p1)'; (H*p2)'; (H*p3)'; (H*p4)'];
    
    % Cartesian coordinates of transformed corners
    p_transf_cartesian = [p_transf_homogeneous(:, 1)./p_transf_homogeneous(:, 3) ...
                            p_transf_homogeneous(:, 2)./p_transf_homogeneous(:, 3)];

    % Upleft and downright corners
    newzero = min(p_transf_cartesian);
    newend = max(p_transf_cartesian);
    
    % Initialization of the destination image
    destination_Image = uint8(zeros(round(newend(1) - newzero(1)), round(newend(2) - newzero(2)), 3));
    size_destination = size(destination_Image);
    
%   Every pixel is backtransformed
for row = 1:size_destination(1)
    for col = 1:size_destination(2)
        p_transf_cart = [row col] + newzero;
%       The pixel is transformed to homogeneous coordinates
        p = double([p_transf_cart 1]');
        ph = H\p;
           
%       Cartesian coordinates are obtained
        pcart(1) = ph(1)/ph(3);
        pcart(2) = ph(2)/ph(3);
        cim = pcart(1);
        rim = pcart(2);
        
%       Rounding so an integer is computed since they are coordinates        
        cc = round(cim);
        rr = round(rim);

%       The pixels are placed on the destination image
%       (First is check the pixel belongs to the original image)
        if  rr > 0 && rr <= size_transposed(1) && cc > 0 && cc <= size_transposed(2)
            destination_Image(row, col, :) = transposed_Im(rr, cc, :);
        end
    end
end
end