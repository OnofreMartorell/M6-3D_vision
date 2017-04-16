%Compute the vanishing points for real images
addpath(genpath('./vanishing_points_v0.4_compiled'));
% pkg load image
% pkg load statistics
img_in1 =  'Data/0000_s.png'; % input image
img_in2 =  'Data/0001_s.png'; % input image
folder_out = '.'; % output folder
manhattan = 1;
acceleration = 0;
focal_ratio = 1;
params.PRINT = 1;
params.PLOT = 1;


% Compute the vanishing points in each image
[horizon1, VPs1] = detect_vps(img_in1, folder_out, manhattan, acceleration, focal_ratio, params);
[horizon2, VPs2] = detect_vps(img_in2, folder_out, manhattan, acceleration, focal_ratio, params);

save -mat7-binary 'VPs_real_images.mat' 'horizon1' 'VPs1' 'horizon2' 'VPs2'
%[horizon01, VPs01] = detect_vps(img_in1, folder_out, manhattan, acceleration, focal_ratio, params);