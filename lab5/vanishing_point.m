function [v1] = vanishing_point(xo1, xf1, xo2, xf2)
% ToDo: create the function 'vanishing_point' that computes the vanishing
% point formed by the line that joins points xo1 and xf1 and the line 
% that joins points x02 and xf2

% Pair of parallel lines
l1 = cross(xo1, xf1);
l2 = cross(xo2, xf2);

% Vanishing point
v1 = cross(l1, l2);
end

