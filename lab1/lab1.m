%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lab 1: Image rectification


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Applying image transformations

% ToDo: create the function  "apply_H" that gets as input a homography and
% an image and returns the image transformed by the homography.
% At some point you will need to interpolate the image values at some points,
% you may use the Matlab function "interp2" for that.


%% 1.1. Similarities
I = imread('Data/0005_s.png'); % we have to be in the proper folder

% ToDo: generate a matrix H which produces a similarity transformation

scaleFactor = 1;
rotationAngle = pi/6;
translationX = 0;
translationY = 0;

H=[scaleFactor * cos(rotationAngle)     scaleFactor * -sin(rotationAngle)    translationX;
   scaleFactor *  sin(rotationAngle)    scaleFactor *  cos(rotationAngle)    translationY;
            0                                    0                                 1     ];

tform = projective2d(H);

close all
I3 = imwarp(I, tform);
I2 = apply_H(I, H);
figure; imshow(I); 
figure; imshow(uint8(I2));
figure; imshow(uint8(I3)); %Uncomment for show imwarp as reference

%% 1.2. Affinities

% ToDo: generate a matrix H which produces an affine transformation
H = [1 0 0;
    tan(-pi/6) 1 0;
    0 0 1];
I2 = apply_H(I, H);
figure; imshow(I); figure; imshow(uint8(I2));

% ToDo: decompose the affinity in four transformations: two
% rotations, a scale, and a translation

% ToDo: verify that the product of the four previous transformations
% produces the same matrix H as above

% ToDo: verify that the proper sequence of the four previous
% transformations over the image I produces the same image I2 as before



%% 1.3 Projective transformations (homographies)

% ToDo: generate a matrix H which produces a projective transformation
theta = 10;
H = [cosd(theta) -sind(theta) 0.001; 
    sind(theta) cosd(theta) 0.01; 
    0 0 1];
tform = projective2d(H);
outputImage = imwarp(I, tform);
figure, imshow(outputImage);
I2 = apply_H(I, H);
figure; imshow(I); figure; imshow(uint8(I2));


%%
A = imread('Data/0000_s.png');
% Create geometric transformation object.

theta = 10;
tform = projective2d([cosd(theta) -sind(theta) 0.001; sind(theta) cosd(theta) 0.01; 0 0 1]);
% Apply transformation and view image.

outputImage = imwarp(A,tform);
figure, imshow(outputImage);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. Affine Rectification


% Choose the image points
I = imread('Data/0000_s.png');
A = load('Data/0000_s_info_lines.txt');

% Indices of lines
i = 424;
p1 = [A(i,1) A(i,2) 1]';
p2 = [A(i,3) A(i,4) 1]';
i = 240;
p3 = [A(i,1) A(i,2) 1]';
p4 = [A(i,3) A(i,4) 1]';
i = 712;
p5 = [A(i,1) A(i,2) 1]';
p6 = [A(i,3) A(i,4) 1]';
i = 565;
p7 = [A(i,1) A(i,2) 1]';
p8 = [A(i,3) A(i,4) 1]';

% ToDo: compute the lines l1, l2, l3, l4, that pass through the different pairs of points
% Done
l1 = cross(p1, p2);
l2 = cross(p3, p4);
l3 = cross(p5, p6);
l4 = cross(p7, p8);

% Show the chosen lines in the image
figure; imshow(I);
hold on;
t = 1:0.1:1000;
plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
plot(t, -(l2(1)*t + l2(3)) / l2(2), 'y');
plot(t, -(l3(1)*t + l3(3)) / l3(2), 'y');
plot(t, -(l4(1)*t + l4(3)) / l4(2), 'y');

% ToDo: compute the homography that affinely rectifies the image

% Vanishing points
v1 = cross(l1, l2);
v2 = cross(l3, l4);

% Line at infinity
l_inf = cross(v1, v2);

l_inf = l_inf/norm(l_inf);
H = [1 0 0; 
    0 1 0; 
    l_inf'];
I2 = apply_H(I, H);
tform = projective2d(H);

I3 = imwarp(I, tform);
% permute(I, [2 1 3])
figure; imshow(uint8(I3));

% ToDo: compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l

H_inv = inv(H)';
lr1 = H_inv*l1;
lr2 = H_inv*l2;
lr3 = H_inv*l3;
lr4 = H_inv*l4;

% show the transformed lines in the transformed image
figure; imshow(uint8(I2));
hold on;
t = 1:0.1:1000;
plot(t, -(lr1(1)*t + lr1(3)) / lr1(2), 'y');
plot(t, -(lr2(1)*t + lr2(3)) / lr2(2), 'y');
plot(t, -(lr3(1)*t + lr3(3)) / lr3(2), 'y');
plot(t, -(lr4(1)*t + lr4(3)) / lr4(2), 'y');

% ToDo: to evaluate the results, compute the angle between the different pair 
% of lines before and after the image transformation

% Angle between line 1 and line 2 before rectification
normal_l1 = [l1(1)/l1(3) l1(2)/l1(3)];
normal_l2 = [l2(1)/l2(3) l2(2)/l2(3)];
norm_l1 = sqrt(dot(normal_l1, normal_l1));
norm_l2 = sqrt(dot(normal_l2, normal_l2));
angle_l1_l2 = acos(dot(normal_l1, normal_l2)/(norm_l1*norm_l2));

% Angle between line 1 and line 2 after rectification
normal_lr1 = [lr1(1)/lr1(3) lr1(2)/lr1(3)];
normal_lr2 = [lr2(1)/lr2(3) lr2(2)/lr2(3)];
norm_lr1 = sqrt(dot(normal_lr1, normal_lr1));
norm_lr2 = sqrt(dot(normal_lr2, normal_lr2));
angle_lr1_lr2 = acos(dot(normal_lr1, normal_lr2)/(norm_lr1*norm_lr2));

% Angle between line 3 and line 4 before rectification
normal_l3 = [l3(1)/l3(3) l3(2)/l3(3)];
normal_l4 = [l4(1)/l4(3) l4(2)/l4(3)];
norm_l3 = sqrt(dot(normal_l3, normal_l3));
norm_l4 = sqrt(dot(normal_l4, normal_l4));
angle_l3_l4 = acos(dot(normal_l3, normal_l4)/(norm_l3*norm_l4));

% Angle between line 3 and line 4 after rectification
normal_lr3 = [lr3(1)/lr3(3) lr3(2)/lr3(3)];
normal_lr4 = [lr4(1)/lr4(3) lr4(2)/lr4(3)];
norm_lr3 = sqrt(dot(normal_lr3, normal_lr3));
norm_lr4 = sqrt(dot(normal_lr4, normal_lr4));
angle_lr3_lr4 = acos(dot(normal_lr3, normal_lr4)/(norm_lr3*norm_lr4));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3. Metric Rectification

%% 3.1 Metric rectification after the affine rectification (stratified solution)

% ToDo: Metric rectification (after the affine rectification) using two non-parallel orthogonal line pairs
%       As evaluation method you can display the images (before and after
%       the metric rectification) with the chosen lines printed on it.
%       Compute also the angles between the pair of lines before and after
%       rectification.

Hm = eye(3);
% Compute the lines l1, m1, l2, m2, that pass through the different pairs of points
l1 = lr1;
m1 = lr2;
l2 = lr3;
m2 = lr4;

% Choose the image points
I = imread('Data/0000_s.png');
A = load('Data/0000_s_info_lines.txt');

% Indices of lines
i = 424;
p1 = [A(i,1) A(i,2) 1]';
p2 = [A(i,3) A(i,4) 1]';
i = 712;
p3 = [A(i,1) A(i,2) 1]';
p4 = [A(i,3) A(i,4) 1]';
i = 565;
p5 = [A(i,1) A(i,2) 1]';
p6 = [A(i,3) A(i,4) 1]';
i = 240;
p7 = [A(i,1) A(i,2) 1]';
p8 = [A(i,3) A(i,4) 1]';

% ToDo: compute the lines l1, l2, l3, l4, that pass through the different pairs of points
% Done
l1 = cross(p1, p2);
m1 = cross(p3, p4);
l2 = cross(p5, p6);
m2 = cross(p7, p8);

% Show the chosen lines in the image
figure;imshow(I);
hold on;
t = 1:0.1:1000;
plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
plot(t, -(m1(1)*t + m1(3)) / m1(2), 'y');
plot(t, -(l2(1)*t + l2(3)) / l2(2), 'g');
plot(t, -(m2(1)*t + m2(3)) / m2(2), 'g');

% Set up the constraints from the orthogonal line pairs
M = [l1(1)*m1(1), l1(1)*m1(2) + l1(2)*m1(1), l1(2)*m1(2);
     l2(1)*m2(1), l2(1)*m2(2) + l2(2)*m2(1), l2(2)*m2(2)];
 
% Find s and S from the null space of the constraints matrix
s = null(M);
S = [s(1) s(2); s(2) s(3)];

% [R,p] = chol(A) for positive definite A, produces an upper triangular matrix
% R from the diagonal and upper triangle of matrix A, satisfying the
% equation R'*R=A and p is zero. If A is not positive definite, then p is
% a positive integer and MATLAB does not generate an error. When A is full,
% R is an upper triangular matrix of order q=p-1 such that R'*R=A(1:q,1:q).
% When A is sparse, R is an upper triangular matrix of size q-by-n so that
% the L-shaped region of the first q rows and first q columns of R'*R agree
% with those of A.

[K,p] = chol(S,'upper');
Hm(1:2,1:2) = K;
Hm = inv(Hm);

% Apply homography
I3 = apply_H(I2, H_m);

% Compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l
H_inv = inv(H)';
l1mr = H_m*l1;
m1mr = H_m*m1;
l2mr = H_m*l2;
m2mr = H_m*m2;

% Show the transformed lines in the transformed image
figure;imshow(uint8(I2));
hold on;
t = 1:0.1:1000;
plot(t, -(l1mr(1)*t + l1mr(3)) / l1mr(2), 'y');
plot(t, -(m1mr(1)*t + m1mr(3)) / m1mr(2), 'y');
plot(t, -(l2mr(1)*t + l2mr(3)) / l2mr(2), 'g');
plot(t, -(m2mr(1)*t + m2mr(3)) / m2mr(2), 'g');

% Evaluate the results, compute the angle between the different pair 
% of lines before and after the image transformation

% Angle between line 1 and line 2 before rectification
normal_l1 = [l1(1)/l1(3) l1(2)/l1(3)];
normal_m1 = [m1(1)/m1(3) m1(2)/m1(3)];
norm_l1 = sqrt(dot(normal_l1, normal_l1));
norm_m1 = sqrt(dot(normal_m1, normal_m1));
angle_l1_m1 = acos(dot(normal_l1, normal_m1)/(norm_l1*norm_m1));

% Angle between line 1 and line 2 after rectification
normal_l1mr = [l1mr(1)/l1mr(3) l1mr(2)/l1mr(3)];
normal_m1mr = [m1mr(1)/m1mr(3) m1mr(2)/m1mr(3)];
norm_l1mr = sqrt(dot(normal_l1mr, normal_l1mr));
norm_m1mr = sqrt(dot(normal_m1mr, normal_m1mr));
angle_l1mr_m1mr = acos(dot(normal_l1mr, normal_m1mr)/(norm_l1mr*norm_m1mr));

% Angle between line 3 and line 4 before rectification
normal_l2 = [l2(1)/l2(3) l2(2)/l2(3)];
normal_m2 = [m2(1)/m2(3) m2(2)/m2(3)];
norm_l2 = sqrt(dot(normal_l2, normal_l2));
norm_m2 = sqrt(dot(normal_m2, normal_m2));
angle_l2_m2 = acos(dot(normal_l2, normal_m2)/(norm_l2*norm_m2));

% Angle between line 3 and line 4 after rectification
normal_l2mr = [l2mr(1)/l2mr(3) l2mr(2)/l2mr(3)];
normal_m2mr = [m2mr(1)/m2mr(3) m2mr(2)/m2mr(3)];
norm_l2mr = sqrt(dot(normal_l2mr, normal_l2mr));
norm_m2mr = sqrt(dot(normal_m2mr, normal_m2mr));
angle_l2mr_m2mr = acos(dot(normal_l2mr, normal_m2mr)/(norm_l2mr*norm_m2mr));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. OPTIONAL: Metric Rectification in a single step
% Use 5 pairs of orthogonal lines (pages 55-57, Hartley-Zisserman book)
% a = [1 2 3];
% b = [4 5 6];
% A = [a; b];
% c = null(A);


% Choose the image points
I = imread('Data/0000_s.png');
A = load('Data/0000_s_info_lines.txt');

% Indices of lines
% Pairs of orthogonal lines
Pairs_lines = [424 712;...
                565 240;...
                534 227;
                576 367;
                424 565];
M = zeros(5, 6);

for k = 1:5
    
    i = Pairs_lines(k, 1);
    p1 = [A(i,1) A(i,2) 1]';
    p2 = [A(i,3) A(i,4) 1]';
    
    i = Pairs_lines(k, 2);
    p3 = [A(i,1) A(i,2) 1]';
    p4 = [A(i,3) A(i,4) 1]';
    
    % Compute pair of orthogonal lines
    l = cross(p1, p2);
    m = cross(p3, p4);
    % Introduce values in the linear system to solve
    M(k, :) = [l(1)*m(1) (l(1)*m(2) + l(2)*m(1))/2 ...
        l(2)*m(2) (l(1)*m(3) + l(3)*m(1))/2 ...
        (l(2)*m(3) + l(3)*m(2))/2, l(3)*m(3)];
end
% Compute the solution of M*c = 0, which is equivalent to find null space
% of M
c = null(M);
% c = (a, b, c, d, e, f)
% Compute C_infinity^*
C_infinity = [c(1) c(2)/2 c(4)/2; ...
            c(2)/2 c(3) c(5)/2; ...
            c(4)/2 c(5)/2 c(6)];
[U, S, V] = svd(C_infinity);
[P, D] = eig(C_infinity);
isequal(U, V')

% H = U;
% H = U*D;

%% 5. OPTIONAL: Affine Rectification of the left facade of image 0000

%% 6. OPTIONAL: Metric Rectification of the left facade of image 0000

%% 7. OPTIONAL: Affine Rectification of the left facade of image 0001

%% 8. OPTIONAL: Metric Rectification of the left facade of image 0001


