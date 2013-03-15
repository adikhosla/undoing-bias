function [model] = learnmodel(data_info, C1, C2, lambda)
%
% Copyright Aditya Khosla http://mit.edu/khosla
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, T. Zhou, T. Malisiewicz, A. Efros, A. Torralba
%   Undoing the Damage of Dataset Bias
%   European Conference on Computer Vision (ECCV) 2012
%   http://undoingbias.csail.mit.edu
%

outputFolder = data_info.outputFolder;
name = data_info.name;

datfile = data_info.datfile;
hdrfile = [outputFolder '/' name '.hdr'];
modfile = [outputFolder '/' name '.mod'];
inffile = [outputFolder '/' name '.inf'];
lobfile = [outputFolder '/' name '.lob'];
cntfile = [outputFolder '/' name '.cnt'];

labelsize = 6;
model.numblocks = 2;
model.numdatasets = data_info.numdatasets;
model.blocksizes = data_info.blocksizes;
model.regmult = [0 1];
model.learnmult = [20 1];
model.lowerbounds{1} = -100;
model.lowerbounds{2} = -100*ones(model.blocksizes(2), 1);
model.hdrfile = hdrfile;
model.datfile = datfile;
model.lobfile = lobfile;

writeheader(hdrfile, data_info.numexamples, labelsize, model);
writelob(lobfile, model)
writecntfile(cntfile, repmat(C2, [model.numdatasets 1]));

fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes) * (model.numdatasets + 1), 1), 'double');
fclose(fid);
fid = fopen(inffile, 'w');
fclose(fid);

cmd = sprintf('./internal/learn %.4f %.4f %.4f %s %s %s %s %s %s', ...
              C1, 1, lambda, hdrfile, datfile, modfile, inffile, lobfile, cntfile);
fprintf('Executing learning code...\n');
status = unix(cmd);

[det_w, det_bias] = readmodel(modfile, model);
w = [det_w{2}; det_w{1}];
bias = cell(model.numdatasets, 1);
for i=1:model.numdatasets
  bias{i} = [det_bias{i}{2}; det_bias{i}{1}];
end

model.w = w;
model.bias = bias;
