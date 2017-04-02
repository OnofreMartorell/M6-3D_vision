function disparity = stereo_computation(left_im,right_im,min_disp,max_disp,w_size,matching_cost)

[row, col] = size(left_im);

% Intermediate matrix to store the result for each pixel depending on the maximum diparity 
pixel_correlation = zeros(row,col,1+max_disp);

for i = ceil(w_size/2):1:row-ceil(w_size/2)
    for j = (max_disp/2)+ceil(w_size/2):1:col-(max_disp/2)-ceil(w_size/2)
        block_left = left_im((i - (ceil(w_size/2)-1)):(i + (ceil(w_size/2)-1)),...
            (j - (ceil(w_size/2)-1)):(j + (ceil(w_size/2)-1)));
        
       ind_disp = 1; 
       for k =j-(max_disp/2):1:j+(max_disp/2)           
           block_right = right_im((i - (ceil(w_size/2)-1)):(i + (ceil(w_size/2)-1)),...
               (k - (ceil(w_size/2)-1)):(k + (ceil(w_size/2)-1)));
          
           if strcmp(matching_cost,'SSD')
                pixel_correlation(i,j,ind_disp) = sum(sum((block_left - block_right).^2));
%            elseif strcmp(matching_cost,'NCC')
%                %pixel_correlation(i,j,ind_disp) = ;
           end
           ind_disp = ind_disp + 1;
       end
    end
end

[~,ind]=min(pixel_correlation,[],3);
disparity = ind-1;
end

