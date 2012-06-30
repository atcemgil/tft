function [tfd_cell] = TFDimensionList2cell(tfd_list)

tfd_cell = cell(length(tfd_list),1);
for i= 1:length(tfd_list)
    tfd_cell{i} = tfd_list(i).name;
end
