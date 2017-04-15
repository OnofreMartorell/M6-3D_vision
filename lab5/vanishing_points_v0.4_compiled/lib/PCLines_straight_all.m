function [u, v] = PCLines_straight(l);
% Transforms line as [x1 y1 x2 y2] or a point as [x y] with PCLines straight
% transform. Coordinates should be normalized.
% see http://medusa.fit.vutbr.cz/public/data/papers/2011-CVPR-Dubska-PClines.pdf


d = 1; % aribitrary distance between vertical axes x and y

L = size(l,2);
if L==4
    % it's a line, convert to point
x1 = l(:,1);
y1 = l(:,2);
x2 = l(:,3);
y2 = l(:,4);

dy = y2-y1;
dx = x2-x1;

m = dy./dx;
b = (y1.*x2 - y2.*x1)./dx;

PCline = [repmat(d,size(b)), b, 1-m]; % homogeneous coordinates

u = PCline(:,1)./PCline(:,3);
v = PCline(:,2)./PCline(:,3);

end

if L==2
    % it's  a point
    x = l(:,1);
    y = l(:,2);
    
    b = x;
    m = (y-x)/d;
    
    u = m;
    v = b;
    
end