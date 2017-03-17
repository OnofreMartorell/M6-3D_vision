function H = homography2d(x1, x2)    
    % Normalization
    [x1n,T1] = normPts(x1);
    [x2n,T2] = normPts(x2);
    
    % Compute DLT
    sizeX1 = size(x1n);
    A = zeros(2*sizeX1(2),9);

    for i=1:2:sizeX1(2)

    A(i:i+1,:) = [0 0 0 -x2n(3,i)*x1n(1,i) -x2n(3,i)*x1n(2,i) -x2n(3,i)*x1n(3,i) x2n(2,i)*x1n(1,i) x2n(2,i)*x1n(2,i) x2n(2,i)*x1n(3,i);
        x2n(3,i)*x1n(1,i) x2n(3,i)*x1n(2,i) x2n(3,i)*x1n(3,i) 0 0 0 -x2n(1,i)*x1n(1,i) -x2n(1,i)*x1n(2,i) -x2n(1,i)*x1n(3,i)];
    end
    
    [~, ~, V] = svd(A);
    H_vector = V(:,end);
    H = reshape(H_vector,[3,3])';
    
    % Desnormalization
    H = inv(T2) * H * T1 ;
end