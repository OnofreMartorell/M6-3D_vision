function [NewPts,T] = normPts(points)

    finiteind = find(abs(points(3,:)) > eps);
    
    if length(finiteind) ~= size(points,2)
        warning('Some points are at infinity');
    end

    points(1,finiteind) = points(1,finiteind)./points(3,finiteind);
    points(2,finiteind) = points(2,finiteind)./points(3,finiteind);
    points(3,finiteind) = 1;

    c = mean(points(1:2, finiteind)')';            % Centroid of finite points
    newp(1,finiteind) = points(1,finiteind)-c(1); % Shift origin to centroid.
    newp(2,finiteind) = points(2,finiteind)-c(2);

    dist = sqrt(newp(1,finiteind).^2 + newp(2,finiteind).^2);
    meandist = mean(dist(:));  % Ensure dist is a column vector for Octave 3.0.1

    scale = sqrt(2)/meandist;

    T = [scale   0   -scale*c(1)
         0     scale -scale*c(2)
         0       0      1      ];

    NewPts = T*points;
end