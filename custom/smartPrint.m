function msgOut = smartPrint(msgIn)
    persistent reverseStr

    if(isempty(reverseStr))
        reverseStr = '';
    end

    msgOut = sprintf([reverseStr, msgIn]);
    reverseStr = repmat(sprintf('\b'), 1, length(msgIn));
end