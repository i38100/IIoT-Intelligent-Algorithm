function bmap = edgeGradient(im)
%执行非最大抑制，返回检测到的边缘
sigma = 3;%暂定
[mag,theta] = gradientMagnitude(im, sigma);
gray = rgb2gray(im);
bmap = edge(gray,"canny") .* mag;
%bmap = nonmax(gray,theta);
end
