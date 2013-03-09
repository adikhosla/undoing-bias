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
datasets = {'data1', 'data2'};
unseenData = 'data3';

% Define feature folder and object category
featuresFolder = 'features/';
object = 'car';

% Set up folders for writing cache files to disk
cacheFolder = ['cache_small/' object '/'];

testName = [object '_' unseenData];
make_dir(cacheFolder);

% Create structures to hold train + test data
Xtrain = cell(length(datasets), 1);
ytrain = cell(length(datasets), 1);

Xtest = cell(length(datasets), 1);
ytest = cell(length(datasets), 1);

n_train = [300 200]; n_test = [500 100];
f = 20;

for i=1:length(datasets)
  % Features are matrices of size n*f where f is the 
  % number of features and n is the number of data points
  Xtrain{i} = rand(n_train(i), f);
  
  % Train and test labels should be n*1 with values {1, -1}
  ytrain{i} = 2*(rand(n_train(i), 1)>0.5)-1;
  
  % Add bias term to test data (added by learning code for training data)
  Xtest{i} = [rand(n_test(i), f) ones(n_test(i), 1)];
  ytest{i} = 2*(rand(n_test(i), 1)>0.5)-1;
end

% Define hyperparameters
C1 = 1000;
lambda = 0.3;
C2 = C1/500;

% Initialize data by writing to disk in required format
%  this is done outside learnmodel to enable faster parameter sweeps
%  by reducing disk I/O
data_info = initdata(Xtrain, ytrain, cacheFolder, testName);

model = learnmodel(data_info, C1, C2, lambda);

% Load unseen dataset
n_unseen = 400;
unseenTest.test_features = [rand(n_unseen, f) ones(n_unseen, 1)];
unseenTest.test_labels = 2*(rand(n_unseen, 1)>0.5)-1;
unseenDecVal = unseenTest.test_features*model.w;
unseenAP = myAP(unseenDecVal, unseenTest.test_labels, 1);
fprintf('\n AP on unseen data (%s): %.4f\n', unseenData, unseenAP);

% Create a table to display results
fprintf('\n AP on seen data (rows: train, columns: test):\n');
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
