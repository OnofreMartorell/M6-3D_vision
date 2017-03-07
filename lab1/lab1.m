%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Lab 1: Image rectification


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1. Applying image transformations

% ToDo: create the function  "apply_H" that gets as input a homography and
% an image and returns the image transformed by the homography.
% At some point you will need to interpolate the image values at some points,
% you may use the Matlab function "interp2" for that.


%% 1.1. Similarities
close all

I = imread('Data/0005_s.png'); % we have to be in the proper folder

% ToDo: generate a matrix H which produces a similarity transformation

scaleFactor = 1;
rotationAngle = pi/6;
translationX = 10;
translationY = -5;

H=[scaleFactor * cos(rotationAngle)     scaleFactor * -sin(rotationAngle)    translationX;
   scaleFactor *  sin(rotationAngle)    scaleFactor *  cos(rotationAngle)    translationY;
            0                                    0                                 1     ];

I2 = apply_H(I, H);     %Handcrafted method
        
tform = projective2d(H); 
%I3 = imwarp(I,tform);   %Matlab method for checking


figure; imshow(I); 
figure; imshow(uint8(I2));
%figure; imshow(uint8(I3)); %Uncomment for show imwarp as reference

%% 1.2. Affinities

% ToDo: generate a matrix H which produces an affine transformation
translationX = 3.2;
translationY = -12;

affine =  [0 1;tan(-pi/6) 1]; %|affine| must be nonzero (affine non singular)

H = [affine(1,1)          affine(1,2)       translationX;
     affine(2,1)          affine(2,2)       translationY;
        0                      0                 1      ];

I2 = apply_H(I, H);

figure; imshow(I); figure; imshow(uint8(I2));

% ToDo: decompose the affinity in four transformations: two
% rotations, a scale, and a translation

%The translation is extracted from the last elements on the first and
%second column. The rotation and the scale factor are extracted from Single
%Value Decomposition. %So U is a rotation, S an anisotropic scaling and V 
%another rotation. The matrix are:

[U,S,V] = svd(affine);  % H = U*S*V' and H = U*V' * V * D * V'

T = [1       0   translationX;
     0       1   translationY;
     0       0         1    ];

R1 = [U(1,1)       U(1,2)        0;
      U(2,1)       U(2,2)        0;
      0             0            1];
 
R2 = [V(1,1)       V(1,2)        0;
      V(2,1)       V(2,2)        0;
      0             0            1];
 
Sc = [S(1,1)       0          0;
     0           S(2,2)       0;
     0             0          1];
 
% ToDo: verify that the product of the four previous transformations
% produces the same matrix H as above

M = T*R1*Sc*R2;

threshold = 1e-10; 
% set your level of accuracy for "equality" since computationally they are
% not "equal"

 if all(abs(H - M)<=threshold)
     disp('Both transformations are the same!')
 else
     disp('The transformations do not match...')
 end
 
% ToDo: verify that the proper sequence of the four previous
% transformations over the image I produces the same image I2 as before

Inew = apply_H(I,M);
figure; imshow(I); figure; imshow(uint8(I2)),title 'Transformed with H';figure; imshow(uint8(Inew)), title 'Transformed with M';
 if isequal(I2,Inew)
     disp('Both transformed images are the same!')
 else
     disp('The transformed images do not match...')
 end
 
Ir2 = apply_H(I,R2);
Is = apply_H(Ir2,Sc);
Ir1 = apply_H(Is,R1);
It = apply_H(Ir1,T);
figure, imshow(uint8(It)), title 'Final waterfall transformed'


%% 1.3 Projective transformations (homographies)

% ToDo: generate a matrix H which produces a projective transformation
H = [0.7        0.1         3;
    0.5         0.3         4;
    0.000003    0.000007    1];

I2 = apply_H(I, H,"linear");
figure; imshow(I); figure; imshow(uint8(I2));


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


I2 = apply_H(permute(I, [2 1 3]), H);

% tform = projective2d(H);

% I3 = imwarp(I, tform);
I2 = permute(I2, [2 1 3]);
% figure; imshow(uint8(I3));


% ToDo: compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l

Hm_inv = inv(H)';
lr1 = Hm_inv*l1;
lr2 = Hm_inv*l2;
lr3 = Hm_inv*l3;
lr4 = Hm_inv*l4;

% show the transformed lines in the transformed image
figure; imshow(uint8(I2));
hold on;
t = 1:0.1:1000;
plot(t, -(lr1(1)*t + lr1(3)) / lr1(2), 'y');
plot(t, -(lr2(1)*t + lr2(3)) / lr2(2), 'g');
plot(t, -(lr3(1)*t + lr3(3)) / lr3(2), 'b');
plot(t, -(lr4(1)*t + lr4(3)) / lr4(2), 'r');

% ToDo: to evaluate the results, compute the angle between the different pair 
% of lines before and after the image transformation

% Angle between line 1 and line 2 before rectification
normal_l1 = [l1(1)/l1(3) l1(2)/l1(3)];
normal_d1 = [l2(1)/l2(3) l2(2)/l2(3)];
norm_l1 = sqrt(dot(normal_l1, normal_l1));
norm_d1 = sqrt(dot(normal_d1, normal_d1));
angle_l1_l2 = acos(dot(normal_l1, normal_d1)/(norm_l1*norm_d1));

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


% Compute the lines l1, m1, l2, m2, that pass through the different pairs of points
l1 = lr1;
m1 = lr2;
l2 = lr3;
m2 = lr4;

x1 = cross(l1, l2);
x2 = cross(l1, m2);
x3 = cross(m1, m2);
x4 = cross(m1, l2);

l1 = lr1;m1 = lr3;

d1 = cross(x1,x3);
d2 = cross(x2,x4);

% Show the chosen lines in the image
figure;imshow(I2);
hold on;
t = 1:0.1:1000;
plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
plot(t, -(m1(1)*t + m1(3)) / m1(2), 'y');
plot(t, -(d1(1)*t + d1(3)) / d1(2), 'g');
plot(t, -(d2(1)*t + d2(3)) / d2(2), 'g');

% Set up the constraints from the orthogonal line pairs
M = [l1(1)*m1(1), l1(1)*m1(2) + l1(2)*m1(1), l1(2)*m1(2);
     d1(1)*d2(1), d1(1)*d2(2) + d1(2)*d2(1), d1(2)*d2(2)];

% Find s and S from the null space of the constraints matrix
s = null(M);
S = [s(1) s(2); s(2) s(3)];
[K, p] = chol(S, 'upper');

Hm = eye(3);
Hm(1:2,1:2) = inv(K);

% Apply homography
I3 = apply_H(permute(I2, [2 1 3]), Hm);
I3 = permute(I3, [2 1 3]);


% Compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l
Hm_inv = inv(Hm)';
l1mr = Hm_inv*l1;
m1mr = Hm_inv*m1;
d1mr = Hm_inv*d1;
d2mr = Hm_inv*d2;

% Show the transformed lines in the transformed image
figure;imshow(uint8(I3));
hold on;
t = 1:0.1:1000;
plot(t, -(l1mr(1)*t + l1mr(3)) / l1mr(2), 'y');
plot(t, -(m1mr(1)*t + m1mr(3)) / m1mr(2), 'y');
plot(t, -(d1mr(1)*t + d1mr(3)) / d1mr(2), 'g');
plot(t, -(d2mr(1)*t + d2mr(3)) / d2mr(2), 'g');

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
normal_d1 = [d1(1)/d1(3) d1(2)/d1(3)];
normal_d2 = [d2(1)/d2(3) d2(2)/d2(3)];
norm_d1 = sqrt(dot(normal_d1, normal_d1));
norm_d2 = sqrt(dot(normal_d2, normal_d2));
angle_d1_d2 = acos(dot(normal_d1, normal_d2)/(norm_d1*norm_d2));

% Angle between line 3 and line 4 after rectification
normal_d1mr = [d1mr(1)/d1mr(3) d1mr(2)/d1mr(3)];
normal_d2mr = [d2mr(1)/d2mr(3) d2mr(2)/d2mr(3)];
norm_d1mr = sqrt(dot(normal_d1mr, normal_d1mr));
norm_d2mr = sqrt(dot(normal_d2mr, normal_d2mr));
angle_d1mr_d2mr = acos(dot(normal_d1mr, normal_d2mr)/(norm_d1mr*norm_d2mr));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4. OPTIONAL: Metric Rectification in a single step
% Use 5 pairs of orthogonal lines (pages 55-57, Hartley-Zisserman book)
% Choose the image points
I = imread('Data/0000_s.png');
A = load('Data/0000_s_info_lines.txt');

% Indices of lines
% Pairs of orthogonal lines
Pairs_lines = [424 712;...
                565 240;...
                534 227;
                576 367];
M = zeros(5, 6);

colors = ['r','g','c','b','y'];
figure;imshow(uint8(I));
l = zeros(3,5);
m = zeros(3,5);
lr = zeros(5,3);
mr = zeros(5,3);
angle_l_m = zeros(5,1);
angle_lr_mr = zeros(5,1);

for k = 1:4
    i = Pairs_lines(k, 1);
    p1 = [A(i, 1) A(i, 2) 1]';
    p2 = [A(i, 3) A(i, 4) 1]';
    
    i = Pairs_lines(k, 2);
    p3 = [A(i, 1) A(i, 2) 1]';
    p4 = [A(i, 3) A(i, 4) 1]';
    
    % Compute pair of orthogonal lines
    l(:, k) = cross(p1, p2);
    m(:, k) = cross(p3, p4);
    % Introduce values in the linear system to solve
    M(k, :) = [l(1,k)*m(1,k) (l(1,k)*m(2,k) + l(2,k)*m(1,k))/2 ...
        l(2,k)*m(2,k) (l(1,k)*m(3,k) + l(3,k)*m(1,k))/2 ...
        (l(2,k)*m(3,k) + l(3,k)*m(2,k))/2, l(3,k)*m(3,k)];
    
    %figure;imshow(uint8(I));
    hold on;
    t = 1:0.1:1000;
    plot(t, -(l(1,k)*t + l(3,k)) / l(2,k), colors(k));
    plot(t, -(m(1,k)*t + m(3,k)) / m(2,k), colors(k));
end
%
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

l1 = cross(p1, p2);
l2 = cross(p3, p4);
l3 = cross(p5, p6);
l4 = cross(p7, p8);

x13 = cross(l1, l3);
x23 = cross(l2, l3);
x24 = cross(l2, l4);
x14 = cross(l1, l4);

k = 5;
l(:, k) = cross(x14, x23);
m(:, k) = cross(x13, x24);
% Introduce values in the linear system to solve
M(k, :) = [l(1,k)*m(1,k) (l(1,k)*m(2,k) + l(2,k)*m(1,k))/2 ...
    l(2,k)*m(2,k) (l(1,k)*m(3,k) + l(3,k)*m(1,k))/2 ...
    (l(2,k)*m(3,k) + l(3,k)*m(2,k))/2, l(3,k)*m(3,k)];

hold on;
t = 1:0.1:1000;
plot(t, -(l(1,k)*t + l(3,k)) / l(2,k), colors(k));
plot(t, -(m(1,k)*t + m(3,k)) / m(2,k), colors(k));


% Compute the solution of M*c = 0, which is equivalent to find null space
% of M
c = null(M);
% Compute C_infinity^*
C_infinity = [c(1) c(2)/2 c(4)/2; ...
            c(2)/2 c(3) c(5)/2; ...
            c(4)/2 c(5)/2 c(6)];
        
[U, S, V] = svd(C_infinity);

[P, D] = eig(C_infinity);
H = P;
% isequal(U, V')
    
% [U, lambda] = eig(C_infinity);
% U_T = U';
% ss = [sqrt(lambda(1)) 0 0; 0 sqrt(lambda(2)) 0; 0 0 0];
% U_T = ss*U_T;
% U = U_T';
% T = inv(U);
% 
% H = U*D;


% Apply homography
I2 = apply_H(permute(I, [2 1 3]), H);
I2 = permute(I2, [2 1 3]);


H = inv(U)';

figure;imshow(uint8(I2));

for k = 1:5
    % Compute the transformed lines -> l'= H^-T*l
    lr(k,:) = H*l(:,k);
    mr(k,:) = H*m(:,k);
    
    %figure;imshow(uint8(I));
    hold on;
    t = 1:0.1:1000;
    plot(t, -(lr(k,1)*t + lr(k,3)) / lr(k,2), colors(k));
    plot(t, -(mr(k,1)*t + mr(k,3)) / mr(k,2), colors(k));
    
    % Angle between pair lines before rectification
    normal_l = [l(1,k)/l(3,k) l(2,k)/l(3,k)];
    normal_m = [m(1,k)/m(3,k) m(2,k)/m(3,k)];
    norm_l = sqrt(dot(normal_l, normal_l));
    norm_m = sqrt(dot(normal_m, normal_m));
    angle_l_m(k,1) = acos(dot(normal_l, normal_m)/(norm_l*norm_m));
    
    % Angle between pair lines after rectification
    normal_lr = [lr(k,1)/lr(k,3) lr(k,2)/lr(k,3)];
    normal_mr = [mr(k,1)/mr(k,3) mr(k,2)/mr(k,3)];
    norm_lr = sqrt(dot(normal_lr, normal_lr));
    norm_mr = sqrt(dot(normal_mr, normal_mr));
    angle_lr_mr(k,1) = acos(dot(normal_lr, normal_mr)/(norm_lr*norm_mr));
end
%% 5. OPTIONAL: Affine Rectification of the left facade of image 0000
%% Affine Rectification
% Choose the image points
I = imread('Data/0000_s.png');
A = load('Data/0000_s_info_lines.txt');

% Indices of lines
i = 493;
p1 = [A(i,1) A(i,2) 1]';
p2 = [A(i,3) A(i,4) 1]';
i = 186;
p3 = [A(i,1) A(i,2) 1]';
p4 = [A(i,3) A(i,4) 1]';
i = 48;
p5 = [A(i,1) A(i,2) 1]';
p6 = [A(i,3) A(i,4) 1]';
i = 508;
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


I2 = apply_H(permute(I, [2 1 3]), H);
I2 = permute(I2, [2 1 3]);

% ToDo: compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l
Hm_inv = inv(H)';
lr1 = Hm_inv*l1;
lr2 = Hm_inv*l2;
lr3 = Hm_inv*l3;
lr4 = Hm_inv*l4;

% show the transformed lines in the transformed image
figure; imshow(uint8(I2));
hold on;
t = 1:0.1:1000;
plot(t, -(lr1(1)*t + lr1(3)) / lr1(2), 'y');
plot(t, -(lr2(1)*t + lr2(3)) / lr2(2), 'g');
plot(t, -(lr3(1)*t + lr3(3)) / lr3(2), 'b');
plot(t, -(lr4(1)*t + lr4(3)) / lr4(2), 'r');

% ToDo: to evaluate the results, compute the angle between the different pair 
% of lines before and after the image transformation

% Angle between line 1 and line 2 before rectification
normal_l1 = [l1(1)/l1(3) l1(2)/l1(3)];
normal_d1 = [l2(1)/l2(3) l2(2)/l2(3)];
norm_l1 = sqrt(dot(normal_l1, normal_l1));
norm_d1 = sqrt(dot(normal_d1, normal_d1));
angle_l1_l2 = acos(dot(normal_l1, normal_d1)/(norm_l1*norm_d1));

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


%% 6. OPTIONAL: Metric Rectification of the left facade of image 0000
%% Metric rectification after the affine rectification (stratified solution)

% ToDo: Metric rectification (after the affine rectification) using two non-parallel orthogonal line pairs
%       As evaluation method you can display the images (before and after
%       the metric rectification) with the chosen lines printed on it.
%       Compute also the angles between the pair of lines before and after
%       rectification.


% Compute the lines that pass through the different pairs of points
l1 = lr1;
m1 = lr2;
l2 = lr3;
m2 = lr4;

x1 = cross(l1, l2);
x2 = cross(l1, m2);
x3 = cross(m1, m2);
x4 = cross(m1, l2);

l1 = lr1;m1 = lr3;

d1 = cross(x1,x3);
d2 = cross(x2,x4);

% Show the chosen lines in the image
figure;imshow(I2);
hold on;
t = 1:0.1:1000;
plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
plot(t, -(m1(1)*t + m1(3)) / m1(2), 'y');
plot(t, -(d1(1)*t + d1(3)) / d1(2), 'g');
plot(t, -(d2(1)*t + d2(3)) / d2(2), 'g');

% Set up the constraints from the orthogonal line pairs
M = [l1(1)*m1(1), l1(1)*m1(2) + l1(2)*m1(1), l1(2)*m1(2);
     d1(1)*d2(1), d1(1)*d2(2) + d1(2)*d2(1), d1(2)*d2(2)];

% Find s and S from the null space of the constraints matrix
s = null(M);
S = [s(1) s(2); s(2) s(3)];
[K, p] = chol(S, 'upper');

Hm = eye(3);
Hm(1:2,1:2) = inv(K);

% Apply homography
I3 = apply_H(permute(I2, [2 1 3]), Hm);
I3 = permute(I3, [2 1 3]);


% Compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l
Hm_inv = inv(Hm)';
l1mr = Hm_inv*l1;
m1mr = Hm_inv*m1;
d1mr = Hm_inv*d1;
d2mr = Hm_inv*d2;

% Show the transformed lines in the transformed image
figure;imshow(uint8(I3));
hold on;
t = 1:0.1:1000;
plot(t, -(l1mr(1)*t + l1mr(3)) / l1mr(2), 'y');
plot(t, -(m1mr(1)*t + m1mr(3)) / m1mr(2), 'y');
plot(t, -(d1mr(1)*t + d1mr(3)) / d1mr(2), 'g');
plot(t, -(d2mr(1)*t + d2mr(3)) / d2mr(2), 'g');

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
normal_d1 = [d1(1)/d1(3) d1(2)/d1(3)];
normal_d2 = [d2(1)/d2(3) d2(2)/d2(3)];
norm_d1 = sqrt(dot(normal_d1, normal_d1));
norm_d2 = sqrt(dot(normal_d2, normal_d2));
angle_d1_d2 = acos(dot(normal_d1, normal_d2)/(norm_d1*norm_d2));

% Angle between line 3 and line 4 after rectification
normal_d1mr = [d1mr(1)/d1mr(3) d1mr(2)/d1mr(3)];
normal_d2mr = [d2mr(1)/d2mr(3) d2mr(2)/d2mr(3)];
norm_d1mr = sqrt(dot(normal_d1mr, normal_d1mr));
norm_d2mr = sqrt(dot(normal_d2mr, normal_d2mr));
angle_d1mr_d2mr = acos(dot(normal_d1mr, normal_d2mr)/(norm_d1mr*norm_d2mr));

%% 7. OPTIONAL: Affine Rectification of the left facade of image 0001
%% Affine Rectification
% Choose the image points
I = imread('Data/0001_s.png');
A = load('Data/0001_s_info_lines.txt');

% Indices of lines
i = 614;
p1 = [A(i,1) A(i,2) 1]';
p2 = [A(i,3) A(i,4) 1]';
i = 159;
p3 = [A(i,1) A(i,2) 1]';
p4 = [A(i,3) A(i,4) 1]';
i = 645;
p5 = [A(i,1) A(i,2) 1]';
p6 = [A(i,3) A(i,4) 1]';
i = 541;
p7 = [A(i,1) A(i,2) 1]';
p8 = [A(i,3) A(i,4) 1]';

% ToDo: compute the lines l1, l2, l3, l4, that pass through the different pairs of points
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


I2 = apply_H(permute(I, [2 1 3]), H);
I2 = permute(I2, [2 1 3]);

% ToDo: compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l

Hm_inv = inv(H)';
lr1 = Hm_inv*l1;
lr2 = Hm_inv*l2;
lr3 = Hm_inv*l3;
lr4 = Hm_inv*l4;

% show the transformed lines in the transformed image
figure; imshow(uint8(I2));
hold on;
t = 1:0.1:1000;
plot(t, -(lr1(1)*t + lr1(3)) / lr1(2), 'y');
plot(t, -(lr2(1)*t + lr2(3)) / lr2(2), 'y');
plot(t, -(lr3(1)*t + lr3(3)) / lr3(2), 'g');
plot(t, -(lr4(1)*t + lr4(3)) / lr4(2), 'g');

% ToDo: to evaluate the results, compute the angle between the different pair 
% of lines before and after the image transformation

% Angle between line 1 and line 2 before rectification
normal_l1 = [l1(1)/l1(3) l1(2)/l1(3)];
normal_d1 = [l2(1)/l2(3) l2(2)/l2(3)];
norm_l1 = sqrt(dot(normal_l1, normal_l1));
norm_d1 = sqrt(dot(normal_d1, normal_d1));
angle_l1_l2 = acos(dot(normal_l1, normal_d1)/(norm_l1*norm_d1));

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

%% 8. OPTIONAL: Metric Rectification of the left facade of image 0001
%% Metric rectification after the affine rectification (stratified solution)

% ToDo: Metric rectification (after the affine rectification) using two non-parallel orthogonal line pairs
%       As evaluation method you can display the images (before and after
%       the metric rectification) with the chosen lines printed on it.
%       Compute also the angles between the pair of lines before and after
%       rectification.


% Compute the lines l1, m1, l2, m2, that pass through the different pairs of points
l1 = lr1;
m1 = lr2;
l2 = lr3;
m2 = lr4;

x1 = cross(l1, l2);
x2 = cross(l1, m2);
x3 = cross(m1, m2);
x4 = cross(m1, l2);

l1 = lr1;m1 = lr3;

d1 = cross(x1,x3);
d2 = cross(x2,x4);

% Show the chosen lines in the image
figure;imshow(I2);
hold on;
t = 1:0.1:1000;
plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
plot(t, -(m1(1)*t + m1(3)) / m1(2), 'y');
plot(t, -(d1(1)*t + d1(3)) / d1(2), 'g');
plot(t, -(d2(1)*t + d2(3)) / d2(2), 'g');

% Set up the constraints from the orthogonal line pairs
M = [l1(1)*m1(1), l1(1)*m1(2) + l1(2)*m1(1), l1(2)*m1(2);
     d1(1)*d2(1), d1(1)*d2(2) + d1(2)*d2(1), d1(2)*d2(2)];

% Find s and S from the null space of the constraints matrix
s = null(M);
S = [s(1) s(2); s(2) s(3)];
[K, p] = chol(S, 'upper');

Hm = eye(3);
Hm(1:2,1:2) = inv(K);

% Apply homography
I3 = apply_H(permute(I2, [2 1 3]), Hm);
I3 = permute(I3, [2 1 3]);


% Compute the transformed lines lr1, lr2, lr3, lr4
%  l'= H^-T*l
Hm_inv = inv(Hm)';
l1mr = Hm_inv*l1;
m1mr = Hm_inv*m1;
d1mr = Hm_inv*d1;
d2mr = Hm_inv*d2;

% Show the transformed lines in the transformed image
figure;imshow(uint8(I3));
hold on;
t = 1:0.1:1000;
plot(t, -(l1mr(1)*t + l1mr(3)) / l1mr(2), 'y');
plot(t, -(m1mr(1)*t + m1mr(3)) / m1mr(2), 'y');
plot(t, -(d1mr(1)*t + d1mr(3)) / d1mr(2), 'g');
plot(t, -(d2mr(1)*t + d2mr(3)) / d2mr(2), 'g');

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
normal_d1 = [d1(1)/d1(3) d1(2)/d1(3)];
normal_d2 = [d2(1)/d2(3) d2(2)/d2(3)];
norm_d1 = sqrt(dot(normal_d1, normal_d1));
norm_d2 = sqrt(dot(normal_d2, normal_d2));
angle_d1_d2 = acos(dot(normal_d1, normal_d2)/(norm_d1*norm_d2));

% Angle between line 3 and line 4 after rectification
normal_d1mr = [d1mr(1)/d1mr(3) d1mr(2)/d1mr(3)];
normal_d2mr = [d2mr(1)/d2mr(3) d2mr(2)/d2mr(3)];
norm_d1mr = sqrt(dot(normal_d1mr, normal_d1mr));
norm_d2mr = sqrt(dot(normal_d2mr, normal_d2mr));
angle_d1mr_d2mr = acos(dot(normal_d1mr, normal_d2mr)/(norm_d1mr*norm_d2mr));

