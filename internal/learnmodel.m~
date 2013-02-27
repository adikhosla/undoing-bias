function [w, bias] = learnmodel(model, C, lambda, modfile, inffile, cntfile)

hdrfile = model.hdrfile;
lobfile = model.lobfile;
datfile = model.datfile;

fid = fopen(modfile, 'wb');
fwrite(fid, zeros(sum(model.blocksizes) * (model.numdatasets + 1), 1), 'double');
fclose(fid);
fid = fopen(inffile, 'w');
fclose(fid);

cmd = sprintf('./learn %.4f %.4f %.4f %s %s %s %s %s %s', ...
              C, 1, lambda, hdrfile, datfile, modfile, inffile, lobfile, cntfile);
fprintf('executing: %s\n', cmd);
status = unix(cmd);

[det_w, det_bias] = readmodel(modfile, model);
w = [det_w{2}; det_w{1}];
bias = cell(model.numdatasets, 1);
for i=1:model.numdatasets
  bias{i} = [det_bias{i}{2}; det_bias{i}{1}];
end
