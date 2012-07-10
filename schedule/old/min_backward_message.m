% sends backward (child to parents) message calculating cumulative
% costs

function [states] = min_backward_message(states, laststate)

done=[];
%min_state_inds=[];
while laststate ~= 1

    for l=1:length(laststate)

        for p=1:length(states(laststate(l)).parents)

            if sum (done == states(laststate(l)).parents(p)) == 0
                costs=[];
                inds=[];
                for c=1:length(states(states(laststate(l)).parents(p)).children)
                    costs = [costs ...
                             states(states(states(laststate(l)) ...
                                           .parents(p)).children(c)).cost ];
                    inds = [inds states(states(laststate(l)).parents(p)).children(c)];
                end

                %min_state_inds = [ min_state_inds inds(find(min(costs)==costs)) ];

                states(states(laststate(l)).parents(p)).cost = ...
                    states(states(laststate(l)).parents(p)).cost + ...
                    min(costs);
                done = [done states(laststate(l)).parents(p)];
            end
        end
    end

    laststate2=[];
    for l=1:length(laststate)
        laststate2 = [laststate2 states(laststate(l)).parents];
    end
    laststate=unique(laststate2);

end
