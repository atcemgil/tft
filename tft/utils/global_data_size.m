function [size] = global_data_size(global_name)
% D1_data and D2_data do not have models but only global data
% this function is written to calculate these data structures size
    eval([ 'global ' global_name ';' ...
           'szs = size(' global_name ');' ]);
    size = 1;
    for i = 1:length(szs)
        size = size * szs(i);
    end
end