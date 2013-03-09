function dispTable = initializeDispTable(datasets)
n = length(datasets);
dispTable = cell(n + 2);
dispTable(2:1+n, 1) = datasets;
dispTable(1, 2:1+n) = datasets;
dispTable{1, end}  = 'Average';
dispTable{end, 1} = 'Visual world';
