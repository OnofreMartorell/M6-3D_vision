%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%    Apply Homgraphy     %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [ destination_Image ] = apply_H( original_Image , H , siz )
%apply_H Performs a specific homography to a given image
%   Inpunt:
%       -Original Image: Image to be transformed
%       -H: matrix 3x3 which defines the homography
%       -siz: size of the destination image
%
%   Output:
%       -Destination Image: Image after homography
%

%   First, an iniziatization of the destination image is done
    destination_Image = uint8(zeros(siz(1), siz(2), 3));
    
%   The homography is called T because maybe further changes will be done
    T = H;
    
%   Every pixel is transform
for row = 1:siz(1)
    for col = 1:siz(2)
        
%       The pixel is transform
        p = double([col row 1]');
        ph = T*p;
        
%       cartesian coordinates are obtained
        pcart(1) = ph(1) / ph(3);
        pcart(2) = ph(2) / ph(3);
        cim = pcart(1);
        rim = pcart(2);
        
%       Rounding so an integer is computed since they are coordinates        
        cc = round(cim);
        rr = round(rim);

%       The pixels are placed on the destination image
        if cc>0 && cc<=size(original_Image, 2) && rr>0 && rr<=size(original_Image, 1)
            destination_Image(row, col, :) = original_Image(rr, cc, :);
        end
    end
end

