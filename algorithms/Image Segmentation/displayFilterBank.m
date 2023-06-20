function displayFilterBank(F)

numFilters = size(F, 3);
filterSize = size(F, 1);

gridSize = ceil(sqrt(numFilters));

for i=1:numFilters
  subplot(gridSize, gridSize, i);
  imagesc(F(:,:,i));
end

% The first 18 filters are first derivatives of Gaussian filters,
% at six different orientations, and three different scales
% (3*6=18).

% The next 18 filters are second derivative of Gaussian filters,
% again at six orientations and three scales.

% The remaining two filters are isotropic Gaussian and LoG filters, 
% each at one scale. 