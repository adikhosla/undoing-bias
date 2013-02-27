function [model] = initdata(Xtrain, ytrain, outputFolder, name)

make_dir([outputFolder '/']);

hdrfile = [outputFolder '/' name '.hdr'];
datfile = [outputFolder '/' name '.dat'];
%modfile = [outputFolder '/' name '.mod'];
%inffile = [outputFolder '/' name '.inf'];
lobfile = [outputFolder '/' name '.lob'];

labelsize = 6;
model.numblocks = 2;
model.numdatasets = length(Xtrain);
model.blocksizes = [1 size(Xtrain{1}, 2)];
model.regmult = [0 1];
model.learnmult = [20 1];
model.lowerbounds{1} = -100;
model.lowerbounds{2} = -100*ones(model.blocksizes(2), 1);
model.hdrfile = hdrfile;
model.datfile = datfile;
model.lobfile = lobfile;

train_features = cell2mat(Xtrain);
train_labels = cell2mat(ytrain);

train_bias = zeros(size(train_features, 1), 1);
currentIdx = 1;
for i=1:length(Xtrain)
  train_bias(currentIdx:currentIdx+size(Xtrain{i}, 1)-1) = i;
  currentIdx = currentIdx + size(Xtrain{i}, 1);
end

num = size(train_features, 1);
writeheader(hdrfile, num, labelsize, model);
writelob(lobfile, model)

fid = fopen(datfile, 'wb');
fclose(fid);

fid = fopen(datfile, 'a');
for i=1:num
  fwrite(fid, [train_labels(i) i 0 0 0 train_bias(i) length(model.blocksizes) sum(model.blocksizes) + length(model.blocksizes)], 'int32');
  fwrite(fid, 1, 'single');
  fwrite(fid, 1.0, 'single');
  fwrite(fid, 2, 'single');
  fwrite(fid, train_features(i, :)', 'single');
end
fclose(fid);
