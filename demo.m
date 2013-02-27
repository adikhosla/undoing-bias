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

setup;

% Set up the seen/unseen datasets
%  You can download the full set of features from:
%  http://undoingbias.csail.mit.edu/features.tar
datasets = {'SUN', 'PASCAL2007'};
unseenData = 'Caltech101';

% Define feature folder and object category
featuresFolder = 'features/';
object = 'car';
testName = [object '_' unseenData];

% Set up folders for writing cache files to disk
cacheFolder = ['cache/' object '/'];
make_dir(cacheFolder);

modfile = [cacheFolder testName '.mod'];
inffile = [cacheFolder testName '.inf'];
cntfile = [cacheFolder testName '.cnt'];

X = cell(length(datasets), 1);
ytrain = cell(length(datasets), 1);

Xtest = cell(length(datasets), 1);
ytest = cell(length(datasets), 1);

for i=1:length(datasets)
  inputFile = [featuresFolder datasets{i} object '_llc_2f.mat'];
  tempData = load(inputFile);
  
  % Features are matrices of size n*f where f is the 
  % number of features and n is the number of data points
  Xtrain{i} = tempData.train_features;
  
  % Train and test labels should be 1*n
  ytrain{i} = tempData.train_labels';
  
  Xtest{i} = [tempData.test_features ones(size(tempData.test_features, 1), 1)];
  ytest{i} = tempData.test_labels';
end

C = 10000;
lambda = 10;
C2 = 100;

model = initdata(Xtrain, ytrain, cacheFolder, testName);
writecntfile(cntfile, repmat(C2, [model.numdatasets 1]));
[w, bias] = learnmodel(model, C, lambda, modfile, inffile, cntfile);

dispTable = cell(length(datasets) + 2);
dispTable(2:1+length(datasets), 1) = datasets;
dispTable(1, 2:1+length(datasets)) = datasets;
dispTable{1, end}  = 'Average';
dispTable{end, 1} = 'All';

for j=1:length(datasets)
    w_i = w + bias{j};
    for k=1:length(datasets)
      decVal = Xtest{k}*w_i;
      dispTable{j+1, k+1} = myAP(decVal, ytest{k}, 1);
    end
end

for j=1:length(datasets)
  decVal = Xtest{j}*w;
  dispTable{end, j+1} = myAP(decVal, ytest{j}, 1); 
end

dispTable(2:end, end) = num2cell(mean(cell2mat(dispTable(2:end, 2:end-1)), 2));
disp(dispTable);
