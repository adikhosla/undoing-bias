function writecntfile(cntfile, cnt, numdatasets)
%
% Copyright Aditya Khosla http://mit.edu/khosla
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, T. Zhou, T. Malisiewicz, A. Efros, A. Torralba
%   Undoing the Damage of Dataset Bias
%   European Conference on Computer Vision (ECCV) 2012
%   http://undoingbias.csail.mit.edu
%

fid = fopen(cntfile, 'wb');
fwrite(fid, cnt, 'int32');
fclose(fid);
