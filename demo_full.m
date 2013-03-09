%
% Copyright Aditya Khosla http://mit.edu/khosla
% 
% This function demonstrates how to use the code base from
% the paper listed below.
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, T. Zhou, T. Malisiewicz, A. Efros, A. Torralba
%   Undoing the Damage of Dataset Bias
%   European Conference on Computer Vision (ECCV) 2012
%   http://undoingbias.csail.mit.edu
%

addpath(genpath('internal'));

% Set up the seen/unseen datasets
%  You can download the full set of features from:
%  http://undoingbias.csail.mit.edu/features.tar
datasets = {'SUN09', 'PASCAL2007'};
unseenData = 'Caltech101';

% Define feature folder and object category
featuresFolder = 'features/';
object = 'car';

% Set up folders for writing cache files to disk
cacheFolder = ['cache/' object '/'];

testName = [object '_' unseenData];
make_dir(cacheFolder);

% Create structures to hold train + test data
Xtrain = cell(length(datasets), 1);
ytrain = cell(length(datasets), 1);

Xtest = cell(length(datasets), 1);
ytest = cell(length(datasets), 1);

for i=1:length(datasets)
  fprintf('Loading dataset (%d of %d): %s\n', i, length(datasets), datasets{i});

  inputFile = [featuresFolder datasets{i} '.mat'];
  tempData = load(inputFile);
  cls_idx = strmatch(object, tempData.classes);
  
  % Features are matrices of size n*f where f is the 
  % number of features and n is the number of data points
  Xtrain{i} = tempData.train_features;
  
  % Train and test labels should be n*1
  ytrain{i} = tempData.train_labels{cls_idx};
  
  Xtest{i} = [tempData.test_features ones(size(tempData.test_features, 1), 1)];
  ytest{i} = tempData.test_labels{cls_idx};
end

% Define hyperparameters
C1 = 10000;
lambda = 1;
C2 = C1/500;

% Initialize data by writing to disk in required format
%  this is done outside learnmodel to enable faster parameter sweeps
%  by reducing disk I/O
data_info = initdata(Xtrain, ytrain, cacheFolder, testName);

model = learnmodel(data_info, C1, C2, lambda);

% Load unseen dataset
fprintf('Loading unseen testdata: %s\n', unseenData);
unseenTest = load([featuresFolder unseenData '.mat'], 'test_features', 'test_labels', 'classes');
unseenDecVal = [unseenTest.test_features ones(size(unseenTest.test_features, 1), 1)] * model.w;
unseenAP = myAP(unseenDecVal, unseenTest.test_labels{strmatch(object, tempData.classes)}, 1);
fprintf('\n AP on unseen dataset (%s): %.4f\n', unseenData, unseenAP);

% Create a table to display results
fprintf('\n AP on seen datasets (rows: train, columns: test):\n');
dispTable = initializeDispTable(datasets);

for j=1:length(datasets)
    w_i = model.w + model.bias{j};
    for k=1:length(datasets)
      decVal = Xtest{k}*w_i;
      dispTable{j+1, k+1} = myAP(decVal, ytest{k}, 1);
    end
end

for j=1:length(datasets)
  decVal = Xtest{j}*model.w;
  dispTable{end, j+1} = myAP(decVal, ytest{j}, 1); 
end

dispTable(2:end, end) = num2cell(mean(cell2mat(dispTable(2:end, 2:end-1)), 2));
disp(dispTable);
