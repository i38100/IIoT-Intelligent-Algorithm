function[mag, theta] = gradientMagnitude(im, sigma)
% 传入图片im和sigma值，进行高斯滤波
%   返回高斯滤波后每个像素点的梯度的大小mag和方向theta
h = fspecial('gaussian',sigma*4+1,sigma);
%为免正负号问题，还是这个d看着舒服
d1 = [-1 1];
d2 = [-1; 1];
%有人说灰度图得到的结果更准确，L2-norm计算没有太便捷的方法，而且结果误差大
%gray = rgb2gray(im)
%我尝试按照任务指导书的方法做

%x方向,%取绝对值
h1 = imfilter(h,d1);
%imx  = imfilter(gray,h1);
imx  = imfilter(im,h1);
imx = computeL2norm(imx);
%imx = abs(imx);
%y方向
h2 = imfilter(h,d2);
%imy  = imfilter(gray,h2);
imy  = imfilter(im,h2);
imy = computeL2norm(imy);
%imy = abs(imy);
%斜率tan值
tantheta = imx./imy;
%theta
theta = atan(tantheta);
mag = sqrt(imx .* imx + imy .* imy);
end