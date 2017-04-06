function lines = run_lsd(img_in);
% Run LSD detector on image
if ~isdeployed
    addpath mex_files/
end

% write image as pgm
img = imread(img_in);
lines = lsd(double(rgb2gray(img)'))'; % uses LSD mex wrapper
lines = lines(:,[1:4]);
