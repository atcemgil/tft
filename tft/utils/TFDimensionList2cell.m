% Helper function to convert dimension list into a cell list with names of dimensions.
function [tfd_cell] = TFDimensionList2cell(tfd_list)

tfd_cell = cell(length(tfd_list),1);
for i= 1:length(tfd_list)
    tfd_cell{i} = tfd_list(i).name;
end
