function [thresh,cntR,sumR,cntP,sumP] = boundaryPRfast(pb,segs,nthresh)
% function [thresh,cntR,sumR,cntP,sumP] = boundaryPRfast(pb,segs,nthresh)
%
% Calcualte precision/recall curve using faster approximation.
% If pb is binary, then a single point is computed with thresh=0.5.
% The pb image can be smaller than the segmentations.
%
% INPUT
%	pb		Soft or hard boundary map.
%	segs		Array of segmentations.
%	[nthresh]	Number of points in PR curve.
%
% OUTPUT
%	thresh		Vector of threshold values.
%	cntR,sumR	Ratio gives recall.
%	cntP,sumP	Ratio gives precision.
%
% See also boundaryPR.
% 
% David Martin <dmartin@eecs.berkeley.edu>
% January 2003

if nargin<3, nthresh = 100; end
if islogical(pb), 
  [thresh,cntR,sumR,cntP,sumP] = boundaryPR(pb,segs,nthresh);
  return
end
nthresh = max(1,nthresh);

[height,width] = size(pb);
nsegs = length(segs);
thresh = linspace(0,1-1/(nthresh),nthresh)';

% compute boundary maps from segs
gt = zeros(size(pb));
bmaps = cell(size(segs));
for i = 1:nsegs,
  bmaps{i} = double(seg2bmap(segs{i},width,height));
  gt = gt + bmaps{i};
end

% thin everything
% for i = 1:nsegs,
%   bmaps{i} = bmaps{i} .* bwmorph(bmaps{i},'thin',inf);
% end

% compute denominator for recall
sumR = 0;
for i = 1:nsegs,
  sumR = sumR + sum(bmaps{i}(:));
end
sumR = sumR .* ones(size(nthresh));
  
% zero counts for recall and precision
cntR = zeros(size(thresh));
cntP = zeros(size(thresh));
sumP = zeros(size(thresh));


m = round(0.005*sqrt(sum(size(bmaps{i}).^2)))*2+1;
pbwide = ordfilt2(pb, m*m, ones(m, m));
gtwide = ordfilt2(gt, m*m, ones(m, m));
scores = cell(size(bmaps));
for i = 1:numel(bmaps)
  scores{i} = pbwide(logical(bmaps{i}));
end
scores = cat(1, scores{:});
%fwrite(2,'[');
for t = 1:nthresh
  %fwrite(2,'.');
  cntR(t) = sum(scores>=thresh(t));
  sumP(t) = sum(pb(:)>=thresh(t));
  cntP(t) = sum(pb(:)>=thresh(t) & gtwide(:)>0);
end
%fprintf(2,']\n');  

return;

fwrite(2,'[');
for t = nthresh:-1:1,
  fwrite(2,'.');
  % threshold and then thin pb to get binary boundary map
  bmap = (pb>=thresh(t));
  %bmap = double(bwmorph(bmap,'thin',inf));
  if t<nthresh,
    % consider only new boundaries
    bmap = bmap .* ~(pb>=thresh(t+1));
    % these stats accumulate
    cntR(t) = cntR(t+1);
    cntP(t) = cntP(t+1);
    sumP(t) = sumP(t+1);
    if ~any(bmap(:))
      continue;
    end    
  end 
  % accumulate machine matches across the human segmentations, since
  % the machine pixels are allowed to match with any segmentation
  accP = zeros(size(pb));
  % compare to each seg in turn
  for i = 1:nsegs,
    
    % compute the correspondence
    dist = bwdist(bmaps{i});
    valid = dist <= 0.01*sqrt(sum(size(bmaps{i}).^2));
    match1 = bmap & valid;
    dist = bwdist(bmap);
    valid = dist <= 0.01*sqrt(sum(size(bmap).^2));
    match2 = bmaps{i} & valid;
    
    %[match1,match2] = correspondPixels(bmap,bmaps{i});
    % compute recall, and mask off what was matched in the groundtruth
    cntR(t) = cntR(t) + sum(match2(:)>0);
    bmaps{i} = bmaps{i} .* ~match2;
    % accumulate machine matches for precision
    accP = accP | match1;
  end
  % compute precision
  sumP(t) = sumP(t) + sum(bmap(:));
  cntP(t) = cntP(t) + sum(accP(:));
end
fprintf(2,']\n');
