function [ disparity ] = plane_sweep(I1, I2, P1, P2, range_depth, size_window, cost_function, step_depth)


length_side_window = ceil(size_window/2) - 1;
sampling_depth = range_depth(1):step_depth:range_depth(2);
[heigth, width] = size(I2);

% In third dimension, first value is for disparity/depth and second for min
% cost
switch cost_function
    case 'SSD'
        disparity_computation = Inf*ones([heigth width 2]);
    case 'NCC'
        disparity_computation = -Inf*ones([heigth width 2]);   
end
for k = 1:length(sampling_depth)
    d = sampling_depth(k);
    PI = P1(3, :) - [0 0 0 d];
    A = pinv([P1; PI]);
    A_hat = A(:, 1:3);
    H = P2*A_hat;
    % Take into account those points that fall inside the area of the other
    % image
    corners = [1 width 1 heigth]; %TODO
    I_reprojected = apply_H_v2(I1, H, corners);
%     uuu = 0;
%     figure,
%     imshow(I_reprojected)
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
            size_block = size(block_right);
            w = (1/(sum(size_block)))*ones(size_block);
            switch cost_function
                case 'SSD'
                    cost_value = sum(sum((w.*(block_left - block_right).^2)));
                    if cost_value < disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = d;
                        disparity_computation(i, j, 2) = cost_value;
                    end
                case 'NCC'
                    num =w.*(block_left - mean2(w.*block_left)).*(block_right - mean2(w.*block_right));
                    den = sqrt(sum(sum(w.*((block_left - mean2(w.*block_left)).^2))))*...
                        sqrt(sum(sum(w.*((block_right - mean2(w.*block_right)).^2))));
                    cost_value = sum(sum(num/den));
                    if cost_value > disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = d;
                        disparity_computation(i, j, 2) = cost_value;
                    end
            end            
        end
    end
end

disparity = disparity_computation(:, :, 1);

end

