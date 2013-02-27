function [blocks, blocks_bias] = readmodel(f, model)

% blocks = readmodel(f, model)
% Read model paramaters from data file.
% Used in the interface with the gradient descent algorithm.

fid = fopen(f, 'rb');
for i = 1:model.numblocks
  blocks{i} = fread(fid, model.blocksizes(i), 'double');
end

blocks_bias = cell(model.numdatasets, 1);

for j=1:model.numdatasets
  for i = 1:model.numblocks
    blocks_bias{j}{i} = fread(fid, model.blocksizes(i), 'double');
  end
end
fclose(fid);
