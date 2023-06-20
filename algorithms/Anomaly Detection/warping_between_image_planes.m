%a function that can take the recovered homography matrix and an image, and return a new image that is the warp of the input image using H
function  warping_between_image_planes(imgA, imgB, H)
    [w, h] = size(imgA);

    %Procedure 1 : obtain the mapping relationship of each pair of points
    img1_x = [0:1:w];
    img1_y = [0:1:h];
    [X, Y] = meshgrid(img1_x, img1_y);
    new_corr = [];
    maxY = 0, maxX = maxY, minY = maxX, minX = minY;
    for i=1:w
        row = Y(i,:);
        for col=1:h
            row_ = insert(row, 2, 1);
            p = transpose(row);
            tmpMatrix = transpose(H*p);
            p_ = [];
            size2 = size(tmpMatrix);
            for j=1:size2
                p_= [p_,tmpMatrix(j,0) / tmpMatrix(j,2), tmpMatrix(j,1) / tmpMatrix(j,2)];
        
                xMax, yMax = max(p_);
                xMin, yMin = min(p_);
        
                minX = min(xMin, minX);
                minY = min(yMin, minY);
                maxX = max(xMax, maxX);
                maxY = max(yMax, maxY);
                new_corr = [new_corr,p_];
            end
        end
    end
    %Procedure 2 : get the mapped picture size
    warpping_h = floor(maxY-minY+2);
    warpping_w = floor(maxX-minX+2);
    result = zeros(warpping_h, warpping_w, 3);
    pix = reshape(imgA, h, w, 3);

    %Procedure 3 : mapping
    [r,c,~]=size(new_corr);
    for i=1:r
        for j=1:c
            corr = result(i,j,:);
            x = floor(corr(0,:)-minX+0.5);
            y = floor(corr(1,:)-minY+0.5);
            result(y,x) = pix(i,j);
        end
    end
    imshow(result);

    %TODO: Procedure 4 :transformed picture after inverse warpping
    

    %TODO: Procedure 5 :merge image

end