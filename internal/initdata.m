function [data_info] = initdata(Xtrain, ytrain, outputFolder, name)
%
% Copyright Aditya Khosla http://mit.edu/khosla
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, T. Zhou, T. Malisiewicz, A. Efros, A. Torralba
%   Undoing the Damage of Dataset Bias
%   European Conference on Computer Vision (ECCV) 2012
%   http://undoingbias.csail.mit.edu
%
make_dir([outputFolder '/']);

data_info.datfile = [outputFolder '/' name '.dat'];
data_info.numdatasets = length(Xtrain);
data_info.numfeatures = size(Xtrain{1}, 2);
data_info.blocksizes = [1 data_info.numfeatures];
data_info.name = name;
data_info.outputFolder = outputFolder;

train_features = cell2mat(Xtrain);
train_labels = cell2mat(ytrain);
data_info.numexamples = size(train_features, 1);

train_bias = zeros(size(train_features, 1), 1);
currentIdx = 1;
for i=1:length(Xtrain)
  train_bias(currentIdx:currentIdx+size(Xtrain{i}, 1)-1) = i;
  currentIdx = currentIdx + size(Xtrain{i}, 1);
end

fid = fopen(data_info.datfile, 'wb');
fclose(fid);

fid = fopen(data_info.datfile, 'a');
for i=1:data_info.numexamples
  fwrite(fid, [train_labels(i) i 0 0 0 train_bias(i) length(data_info.blocksizes) sum(data_info.blocksizes) + length(data_info.blocksizes)], 'int32');
  fwrite(fid, 1, 'single');
  fwrite(fid, 1.0, 'single');
  fwrite(fid, 2, 'single');
  fwrite(fid, train_features(i, :)', 'single');
end
fclose(fid);
