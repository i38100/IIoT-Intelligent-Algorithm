%This is the main procedures in the assignment

%read image
imgA = imread('uttower2.jpg');
imgB = imread('uttower1.jpg');

%choose whether to manually collect point
mode = input('Choosing point manually? default is False. Type 1(true) or 0 (false)');
%initialize H
H1 = zeros(3, 3);
H2 = zeros(3, 3);
if mode
    %get homography parameters manually
    [ptA, ptB] = getting_correspondences(imgA, imgB, 8)
    %Compute transformation (homography) between second image and first using corresponding points
    H1 = computing_homography_parameters(ptA, ptB);
    verify_homography_matrix(imgA, imgB,  H1)
    warping_between_image_planes(imgA, imgB, H1)
else
    %TODO:ptA, ptB = getting_correspondences_sift(imgA, imgB);
    warping_between_image_planes(imgA, imgB, H2)
end

