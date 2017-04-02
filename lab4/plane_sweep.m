function [ disparity ] = plane_sweep(I1, I2, P1, P2, range_disp, size_window, cost_function)

step_disparity = 1;

sampling_depth = range_disp(1):step_disparity:range_disp(2);
[heigth, width] = size(I2);
% In third dimension, first value is for disparity/depth and second for min
% cost
disparity_computation = zeros([heigth width 2]);
for k = 1:length(sampling_depth)
    d = sampling_depth(k);
    PI = P1(3, :) - [0 0 0 d];
    A = pinv([P1; PI]);
    A_hat = A(:, 1:3);
    H = P2*A_hat;
    % Take into account those points that fall inside the area of the other
    % image
    corners = [0 0]; %TODO
    I_reprojected = apply_H_v2(I1, H, corners);
    for i = 1:heigth
        for j = 1:width
            
            switch cost_function               
                case 'ssd'
                    cost_value = 0;
                case 'ncc'
                    cost_value = 0;
            end
            if cost_value < disparity_computation(i, j, 2)               
                disparity_computation(i, j, 1) = d;
            end    
        end    
    end    
end

disparity = disparity_computation(:, :, 1);

end

