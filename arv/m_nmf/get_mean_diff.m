function diff = get_mean_diff(target, current)
    diff = abs(target-current);
    for i=1:ndims(target)
        diff=sum(diff);
    end
    diff = diff / numel(target);
end