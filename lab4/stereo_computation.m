function disparity = stereo_computation(left_im, right_im, min_disp, max_disp, w_size, matching_cost)

[row, col] = size(left_im);
length_side_window = ceil(w_size/2) - 1;
% Intermediate matrix to store the result for each pixel depending on the maximum diparity 
pixel_correlation = zeros(row, col, 1 + max_disp);
% For each point in the left image
for i = ceil(w_size/2):row - ceil(w_size/2)
    for j = (max_disp/2) + ceil(w_size/2):col - (max_disp/2) - ceil(w_size/2)
        % Window on the left image
        block_left = left_im((i - length_side_window):(i + length_side_window),...
            (j - length_side_window):(j + length_side_window));
       % Find the ssd cost with windows at the other image 
       ind_disp = 1; 
       for k = j - (max_disp/2):j + (max_disp/2)           
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
    [~, ind] = min(pixel_correlation,[],3);
 elseif strcmp(matching_cost, 'NCC')
    [~, ind] = max(pixel_correlation,[],3);
end
disparity = ind - 1;

end
