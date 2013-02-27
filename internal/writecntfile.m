function writecntfile(cntfile, cnt, numdatasets)

% writeheader(file, num, labelssize, model)
% Write training header file.
% Used in the interface with the gradient descent algorithm.

fid = fopen(cntfile, 'wb');
fwrite(fid, cnt, 'int32');
fclose(fid);
