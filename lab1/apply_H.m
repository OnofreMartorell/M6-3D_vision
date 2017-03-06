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
    
%   The homography is called T because maybe further changes will be done
    T0 = (H);
    
%   The corners are localized and transformed so the destination size 
%   may be computed    
    
    p1 = double([1,1,1]');
    p2 = double([1,siz(2),1]');
    p3 = double([siz(1), 1,1]');
    p4 = double([siz(1), siz(2),1]');
    
    ph1 = T0*p1;
    ph2 = T0*p2;
    ph3 = T0*p3;
    ph4 = T0*p4;
    
    ph1c = [ph1(1) / ph1(3),  ph1(2) / ph1(3)];
    ph2c = [ph2(1) / ph2(3),  ph2(2) / ph2(3)];
    ph3c = [ph3(1) / ph3(3),  ph3(2) / ph3(3)]; 
    ph4c = [ph4(1) / ph4(3),  ph4(2) / ph4(3)];    
 
    newzeroX = min([ph1c(1),ph2c(1),ph3c(1),ph4c(1)]);
    newzeroY = min([ph1c(2),ph2c(2),ph3c(2),ph4c(2)]);
    
    newendX = max([ph1c(1),ph2c(1),ph3c(1),ph4c(1)]);
    newendY = max([ph1c(2),ph2c(2),ph3c(2),ph4c(2)]);
    
    %  An iniziatization of the destination image is done
    destination_Image = uint8(zeros(round(newendX-newzeroX), round(newendY-newzeroY), 3));
    s = size(destination_Image);
    
%      T1= T0 +  [ 0     0     round(newzeroX);
%                    0     0   round(newzeroY);
%                    0     0     0];
    
    T = inv(T0);
        
    
%   Every pixel is backtransform
for row = 1:s(1)
    for col = 1:s(2)
        
%       The pixel is transform
        p = double([col row 1]');
        ph = T*p;
           
%       cartesian coordinates are obtained
        pcart(1) = ph(1) / ph(3);
        pcart(2) = ph(2) / ph(3);
        cim = pcart(1) + newzeroX;
        rim = pcart(2) + newzeroY;
        
%       Rounding so an integer is computed since they are coordinates        
        cc = round(cim);
        rr = round(rim);

%       The pixels are placed on the destination image
%       (First is check the pixel belongs to the original image)
        if cc>0 && cc<=size(original_Image, 2) && rr>0 && rr<=size(original_Image, 1)
            destination_Image(row, col , :) = original_Image(rr, cc, :);
        end
    end
end

