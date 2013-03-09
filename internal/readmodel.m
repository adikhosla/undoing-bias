function [blocks, blocks_bias] = readmodel(f, model)
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
