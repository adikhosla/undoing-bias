function dispTable = initializeDispTable(datasets)
%
% Copyright Aditya Khosla http://mit.edu/khosla
%
% Please cite this paper if you use this code in your publication:
%   A. Khosla, T. Zhou, T. Malisiewicz, A. Efros, A. Torralba
%   Undoing the Damage of Dataset Bias
%   European Conference on Computer Vision (ECCV) 2012
%   http://undoingbias.csail.mit.edu
%
n = length(datasets);
dispTable = cell(n + 2);
dispTable(2:1+n, 1) = datasets;
dispTable(1, 2:1+n) = datasets;
dispTable{1, end}  = 'Average';
dispTable{end, 1} = 'Visual world';
