function [v1] = vanishing_point(xo1, xf1, xo2, xf2)
% ToDo: create the function 'vanishing_point' that computes the vanishing
% point formed by the line that joins points xo1 and xf1 and the line 
% that joins points x02 and xf2

% Pair of parallel lines
l1 = cross(xo1, xf1);
l1 = l1/l1(end);
l2 = cross(xo2, xf2);
l2 = l2/l2(end);

% Vanishing point
v1 = cross(l1, l2);
end
% x1 = [xo1, xf1, xo2, xf2];
% t = 1:0.1:1000;
% x1e = euclid(x1);
% figure,
% scatter(x1e(1,1:4), x1e(2,1:4))
% hold on
% plot(t, -(l1(1)*t + l1(3)) / l1(2), 'y');
% plot(t, -(l2(1)*t + l2(3)) / l2(2), 'r');
