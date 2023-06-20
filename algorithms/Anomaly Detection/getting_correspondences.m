%Procedure 1: Take a sequence of images from the same position
function [ptA, ptB] = getting_correspondences(imgA, imgB, numPoints)
%input: two images imgA and imgB
%return: homography parameters ptA amd ptB
    subplot(1, 2, 1);
    imshow(imgA);
    title("Click points in A");
    subplot(1, 2, 2);
    imshow(imgB);
    title("Click points in B");

    %please iput in the order of ABABABAB...
    [x,y] = ginput(numPoints);
    

    %seperate ptA and ptB
    ptA = [];
    ptB = [];
    for i=1:floor(numPoints/2)
        ptA = [ptA,[x(2*i-1),y(2*i-1)]];
        ptB = [ptB,[x(2*i),y(2*i)]];
    end
    ptA=transpose(ptA);
    ptA=reshape(ptA,2,4);
    ptA=transpose(ptA);
    ptB=transpose(ptB);
    ptB=reshape(ptB,2,4);
    ptB=transpose(ptB);
end
