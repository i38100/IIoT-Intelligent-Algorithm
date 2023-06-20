% evaluateSegmentation.m

% set the directories for images, ground truth, and results
maindir = './'; % this should point to your prob_seg directory
ids = load(fullfile(maindir, 'iids_test.txt'));
imdir = fullfile(maindir, 'data', 'images');
gtdir = fullfile(maindir, 'data', 'human');
resultbasedir = fullfile(maindir, 'results');
addpath('./util');
addpath('./solutions'); % change to your solutions directory

makehtml = true; % optional, for creating html pages for results
overwrite = true; % set to false if you want to skip already processed images

methods = {'gradient', 'oriented'};

for m = 1:numel(methods) % choose which methods to run
  
  close all;
  
  method = methods{m};
  resultdir = fullfile(resultbasedir, 'color', method);
  if ~exist(resultdir, 'file')
    mkdir(resultdir); % create the result directory for the current method 
  end

  for f = 1:numel(ids) % loop through each test image

    if ~overwrite && exist(fullfile(resultdir, [num2str(ids(f)) '.bmp']), 'file')
     continue; % skip if the output already exists
    end
    
    imfn = fullfile(imdir, [num2str(ids(f)) '.jpg']);
    im = im2double(imread(imfn));
    switch method
      case 'gradient'
        bmap = edgeGradient(im); % you write this
      case 'oriented'
        bmap = edgeOrientedFilters(im); % you write this
      otherwise 
        error(['invalid case: ' method]);
    end
    figure(1), imshow(im); % show input image
    figure(2), imshow(mat2gray(bmap)), axis image, colormap gray % show boundary scores with wider edges
    drawnow; % makes sure that display updates inside loop
    imwrite(bmap, fullfile(resultdir, [num2str(ids(f)) '.bmp'])); % save boundary scores
  end

  boundaryBench(resultdir, ids, gtdir, 100, 1); % do not change the last three params
  boundaryBenchGraphs(resultdir, ids);
  scores_individual = load(fullfile(resultdir, 'scores.txt'));
  scores_overall = load(fullfile(resultdir, 'score.txt'));
  fprintf('Method %s:\toverall F-score = %0.3f\t\taverage F-score = %0.3f\n', ...
    method, scores_overall(4), mean(scores_individual(:, 5))); 
end

% use these lines to create htmls for comparison (optional)
if makehtml
  boundaryBenchHtml(fullfile(maindir, 'results'), ids);
  boundaryBenchGraphsMulti(fullfile(maindir, 'results'), ids);
end