function H = computing_homography_parameters(ptA, ptB)
%input:homography parameters ptA amd ptB
%output: associated 3 × 3 homography matrix H
%a simplify way to compute homography parameters given in the assignment
[row,~]=size(ptA);
numPoints = row*2;
% get 2n×8 matrix A
A = zeros(numPoints, 8);
for i=1:floor(numPoints/2)
    A(2*i-1,:) = [ptA(i,1), ptA(i,2), 1, 0, 0, 0, -ptA(i,1)*ptB(i,1), -ptA(i,2)*ptB(i,1)];
    A(2*i,:) = [0, 0, 0, ptA(i,1), ptA(i,2), 1, -ptA(i,1)*ptB(i,2), -ptA(i,2)*ptB(i,2)];
end
%get vector b   
B = reshape(ptB, numPoints, 1);
%Solve for the unknown homography matrix parameters
H_ = A\ B;
%Let H3,3 = 1, and reshape
H = insert(H_, 9, 1.0);
H = reshape(H,3,3);