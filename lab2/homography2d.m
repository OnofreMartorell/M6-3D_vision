function H = homography2d(x1, x2)

sizeX1 = size(x1);
A=zeros(2*sizeX1(1),9);

for i=1:2:sizeX1(1)
    A(i,:) = [0 0 0 -x2(i,3)*x1(i,1) -x2(i,3)*x1(i,2) -x2(i,3)*x1(i,3) x2(i,2)*x1(i,1) x2(i,2)*x1(i,2) x2(i,2)*x1(i,3);
    x2(i,3)*x1(i,1) x2(i,3)*x1(i,2) x2(i,3)*x1(i,3) 0 0 0 -x2(i,1)*x1(i,1) -x2(i,1)*x1(i,2) -x2(i,1)*x1(i,3)];
end

[U,D,V] = svd(A);
H_vector = V(:,end);

H=reshape(H_vector,[3,3])';

end