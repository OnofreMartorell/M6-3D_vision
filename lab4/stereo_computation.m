function disparity = stereo_computation(left_im, right_im, min_disp, max_disp, w_size, matching_cost)

[row, col] = size(left_im);
length_side_window = ceil(w_size/2) - 1;
switch matching_cost
    case 'SSD'
        disparity_computation = Inf*ones([row col 2]);
    case 'NCC'
        disparity_computation = -Inf*ones([row col 2]);
        
end
% For each point in the left image
for i = length_side_window + 1:row - length_side_window
    for j = length_side_window + 1:col - length_side_window
        % Window on the left image
        block_left = left_im((i - length_side_window):(i + length_side_window),...
            (j - length_side_window):(j + length_side_window));
        % Find the ssd or ncc cost with windows at the other image
        for k = max( j - max_disp - length_side_window, 1 + length_side_window):...
                min(col - length_side_window, j + max_disp - length_side_window)
            
            block_right = right_im((i - length_side_window):(i + length_side_window),...
                (k - length_side_window):(k + length_side_window));
            switch matching_cost
                case 'SSD'
                    cost_value = sum(sum((block_left - block_right).^2));
                    if cost_value < disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = j - k;
                        disparity_computation(i, j, 2) = cost_value;
                    end
                case 'NCC'
                    num =(block_left - mean2(block_left)).*(block_right - mean2(block_right));
                    den = sqrt(sum(sum((block_left - mean2(block_left)).^2)))*sqrt(sum(sum((block_right - mean2(block_right)).^2)));
                    cost_value = sum(sum(num/den));
                    if cost_value > disparity_computation(i, j, 2)
                        disparity_computation(i, j, 1) = j - k;
                        disparity_computation(i, j, 2) = cost_value;
                    end
            end
        end
    end
end
disparity = disparity_computation(:, :, 1);
end
