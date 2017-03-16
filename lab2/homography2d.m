function H = homography2d(x1, x2)

sizeX1 = size(x1);
A=zeros(2*sizeX1(2),9);

for i=1:2:sizeX1(2)
    
A(i:i+1,:) = [0 0 0 -x2(3,i)*x1(1,i) -x2(3,i)*x1(2,i) -x2(3,i)*x1(3,i) x2(2,i)*x1(1,i) x2(2,i)*x1(2,i) x2(2,i)*x1(3,i);
    x2(3,i)*x1(1,i) x2(3,i)*x1(2,i) x2(3,i)*x1(3,i) 0 0 0 -x2(1,i)*x1(1,i) -x2(1,i)*x1(2,i) -x2(1,i)*x1(3,i)];
end

%     A(i,:) = [0 0 0 -x2(i,3)*x1(i,1) -x2(i,3)*x1(i,2) -x2(i,3)*x1(i,3) x2(i,2)*x1(i,1) x2(i,2)*x1(i,2) x2(i,2)*x1(i,3);
%     x2(i,3)*x1(i,1) x2(i,3)*x1(i,2) x2(i,3)*x1(i,3) 0 0 0 -x2(i,1)*x1(i,1) -x2(i,1)*x1(i,2) -x2(i,1)*x1(i,3)];

[U,D,V] = svd(A);
H_vector = V(:,end);

H=reshape(H_vector,[3,3])';

end