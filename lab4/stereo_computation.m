function disparity = stereo_computation(left_im, right_im, min_disp, max_disp, w_size, matching_cost)

[row, col] = size(left_im);
length_side_window = ceil(w_size/2) - 1;
% For each point in the left image
% for i = ceil(w_size/2):row - ceil(w_size/2)
%     for j = ceil(w_size/2):col - ceil(w_size/2)
for i = length_side_window + 1:row - length_side_window 
    for j = length_side_window + 1:col - length_side_window 
        % Window on the left image
        block_left = left_im((i - length_side_window):(i + length_side_window),...
            (j - length_side_window):(j + length_side_window));
        % Find the ssd or ncc cost with windows at the other image
        ind_disp = 1;
        %         for k = j - max(max_disp + length_side_window, j - 1 - length_side_window):...
        %                j + min(col - j - length_side_window, max_disp - length_side_window)
        for k = max( j - max_disp - length_side_window, 1 + length_side_window):...
                min(col - length_side_window, j + max_disp - length_side_window)
            
            block_right = right_im((i - length_side_window):(i + length_side_window),...
                (k - length_side_window):(k + length_side_window));
            
            if strcmp(matching_cost,'SSD')
                pixel_correlation(i, j, ind_disp) = sum(sum((block_left - block_right).^2));
            elseif strcmp(matching_cost, 'NCC')
                num =(block_left - mean2(block_left)).*(block_right - mean2(block_right));
                den = sqrt(sum(sum((block_left - mean2(block_left)).^2)))*sqrt(sum(sum((block_right - mean2(block_right)).^2)));
                pixel_correlation(i, j, ind_disp) = sum(sum(num/den));
                
            end
            ind_disp = ind_disp + 1;
        end
    end
end


if strcmp(matching_cost, 'SSD')
    [~, ind] = min(pixel_correlation, [], 3);
elseif strcmp(matching_cost, 'NCC')
    [~, ind] = max(pixel_correlation, [], 3);
end
disparity = ind - 1;
end
