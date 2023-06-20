function verify_homography_matrix(imgA, imgB, H)
    %verify by mapping the clicked image points from one view to the other
    subplot(1, 2, 1)
    imshow(imgA)
    hold on    
    subplot(1, 2, 2)
    imshow(imgB)
    hold on
    flag = input('Now start to verify. Click a point, and see whether the corresponding point is right. Type 0 to break.');
    while flag
        pts = ginput(1); %print
        pts = reshape(pts, 1 ,2);
        subplot(1, 2, 1)
        %changer the Markersize, it is too small before
        plot(pts(1), pts(2),'--p','MarkerSize',20)
        hold on
        toTrans = transpose([pts(1), pts(2), 1]);
        p = transpose(H * toTrans);
        x = p(1) / p(3);
        y = p(2) / p(3);
        subplot(1, 2, 2)
        plot(x, y,'--p','MarkerSize',20)
        hold on
        flag = input('Continue to verify? 0 to break.');
    end
end