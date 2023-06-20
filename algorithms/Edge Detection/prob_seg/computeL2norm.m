function gradient = computeL2norm(matrix)
%输入的是图片在一个方向的梯度分量，包含RGB三个通道，输出L2正则化后的梯度
R = matrix(:,:,1);
G = matrix(:,:,2);
B = matrix(:,:,3);
gradient = sqrt(R .* R + G .* G + B .* B);
end