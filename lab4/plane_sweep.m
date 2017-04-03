function [ disparity ] = plane_sweep(I1, I2, P1, P2, range_depth, size_window, cost_function)

step_disparity = 1;
length_side_window = ceil(size_window/2) - 1;
sampling_depth = range_depth(1):step_disparity:range_depth(2);
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
    corners = [0 heigth 0 width]; %TODO
    I_reprojected = apply_H_v2(I1, H, corners);
    %For each point in the image
    for i = 1:heigth
        for j = 1:width
            block_left  = I2(max(i - length_side_window, 1 + length_side_window):...
                min(i + length_side_window, heigth - length_side_window),...
            max(j - length_side_window, 1 + length_side_window):...
            min(j + length_side_window, width - length_side_window));
        
            block_right  = I_reprojected(max(i - length_side_window, 1 + length_side_window):...
                min(i + length_side_window, heigth - length_side_window),...
            max(j - length_side_window, 1 + length_side_window):...
            min(j + length_side_window, width - length_side_window));
            switch cost_function
                case 'SSD'
                    cost_value = sum(sum((block_left - block_right).^2));
                    if cost_value < disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = d;
                    end
                case 'NCC'
                    num =(block_left - mean2(block_left)).*(block_right - mean2(block_right));
                    den = sqrt(sum(sum((block_left - mean2(block_left)).^2)))*sqrt(sum(sum((block_right - mean2(block_right)).^2)));
                    cost_value = sum(sum(num/den));
                    if cost_value > disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = d;
                    end
            end            
        end
    end
end

disparity = disparity_computation(:, :, 1);

end
