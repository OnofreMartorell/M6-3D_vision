function F = fundamental_matrix(x1, x2)

[x1_norm, T1] = normalize_points(x1);
[x2_norm, T2] = normalize_points(x2);


x1_norm = euclid(x1_norm);
x2_norm = euclid(x2_norm);

[~, num_pairs] = size(x1);
W = zeros(num_pairs, 9);


for i = 1:num_pairs
    
    W(i, :) = [ x1_norm(1, i)*x2_norm(1, i)
                x1_norm(2, i)*x2_norm(1, i)
                x2_norm(1, i)
                x1_norm(1, i)*x2_norm(2, i)
                x1_norm(2, i)*x2_norm(2, i)
                x2_norm(2, i)
                x1_norm(1, i)
                x1_norm(2, i)
                1 ];
        
end
[~, ~, V] = svd(W);
f = V(:, end);
F_rank3 = reshape(f, [3, 3]);
[U_f, D_f, V_f] = svd(F_rank3);
D_f(3, 3) = 0;

F_rank2 = U_f*D_f*V_f;
F = T2'*F_rank2*T1;



% for i = 1:num_pairs
%     
%     W(i, :) = [ x2_norm(1, i)*x1_norm(1, i)
%                 x2_norm(2, i)*x1_norm(1, i)
%                 x1_norm(1, i)
%                 x2_norm(1, i)*x1_norm(2, i)
%                 x2_norm(2, i)*x1_norm(2, i)
%                 x1_norm(2, i)
%                 x2_norm(1, i)
%                 x2_norm(2, i)
%                 1 ];
%         
% end
% [~, ~, V] = svd(W);
% f = V(:, end);
% F_rank3 = reshape(f, [3, 3]);
% [U_f, D_f, V_f] = svd(F_rank3);
% D_f(3, 3) = 0;
% 
% F_rank2 = U_f*D_f*V_f;
% F = T1'*F_rank2*T2;
end