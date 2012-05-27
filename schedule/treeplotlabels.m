% plots given labels on treeplot images

function [] = treeplotlabels(p, labels)

% from
% http://stackoverflow.com/questions/10148419/treeplot-string-labeling-matlab

treeplot(p);

c = get(gca, 'Children'); % get handles to children
                          % grab X and Y coords from the second
                          % child (the first one is axes)
x = get(c(2), 'XData');
y = get(c(2), 'YData');
text(x + 0.02, y, labels); %, 'VerticalAlignment','bottom', ...
                   %      'HorizontalAlignment','right')