% tensor test set
% generates results of all possible factorizations of two [3 3 3] tensors

% cardinality of 3 indicates tensors have 3 elements in the corresponding
% dimension.

% cardinality of 0 indicates 
%    - contraction on corresponding dimension for the result tensor
%    - no data is available on corresponding dimension for the input tensor

% all possible factorization of two [3 3 3] tensors is described as such:
% maximum cardinality:
% A [3 3 3] * B [3 3 3] = C [3 3 3]
% this case reduces to the hadamard product

% We could generate multiplications of smaller cardinalities such as:
% A [3 3 3] * B [3 3 3] = C [3 3 0]
% in this case we have to sum up corresponding elements of the complete 
% product vector to calculate the result tensor.

% We could move the 0 cardinality to any index of any tensor object in the
% above mentioned multiplication operation.

% We could also have two 0 cardinality dimensions in this factorization.
% Any tensor with two 0 cardinality dimensions would reduce to a vector in
% this case.

% The following code generates all possible combinations of cardinalities
% for one and two 0 cardinality tensors. Results are stored and published
% as a test set.

% if input opnumber is specified, only numbers in the opnumber vector will
% be processed

function [] = tts(opnumber)

    % by specifying TEST=1 and a test folder, can perform checks

    TEST=0;
    TEST_FOLDER='/home/can2/arastir/drtez2/tts/set3/'; % do not forget the last /

    % by specifying OUTPUT=1 and an output folder, can store generated results
    % (overwrites)

    OUTPUT=0;
    OUTPUT_FOLDER='/home/can2/arastir/drtez2/tts/theset/'; % do not forget the last /

    % defines output verbosity
    VERBOSITY=1;

    
    
    
    
    
    
    
    
    
    




    if TEST==1
        display('test mode:');
        display(['sourcing test set from folder: ' TEST_FOLDER]);
    end

    if OUTPUT==1
        display(['storing output to folder: ' OUTPUT_FOLDER]);
    end


    d=3;
    A_card=[unique(perms([d 0 d]), 'rows'); unique(perms([d 0 0]), 'rows')];
    B_card=A_card;
    C_card=B_card;
    
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);

    A=round(rand(d,d,d)*5);
    B=round(rand(d,d,d)*5);

    if VERBOSITY>1
        display(A);
        display(B);
    end

    opnum=1;
    full_result=0;
    
    for a_c=1:size(A_card,1)
        for b_c=1:size(B_card,1)
            for c_c=1:size(C_card,1)
                
                if nargin==1 && sum( opnumber == opnum ) == 0
                    opnum = opnum + 1;
                else

                    if VERBOSITY>0
                        display(['op ' num2str(opnum) ': ' ...
                                 ' A_card ' num2str(A_card(a_c,:)) ...
                                 ' B_card ' num2str(B_card(b_c,:)) ...
                                 ' C_card ' num2str(C_card(c_c,:)) ])
                    end

   % perform the operation
   [op, full_result] = ...
     tensormul( get_tensor_part(A,A_card(a_c,:),VERBOSITY),A_card(a_c,:), ...
                get_tensor_part(B,B_card(b_c,:),VERBOSITY),B_card(b_c,:), ...
                C_card(c_c,:), ...
                VERBOSITY , full_result );


                    if VERBOSITY>0
                        display(op);
                    end


                    if TEST==1
                        test=load([TEST_FOLDER 'op' num2str(opnum)]);
                        s=test.op ~= op;
                        ndims=length(A_card);
                        for dim=1:ndims
                            s=sum(s);
                        end
                        if s ~= 0
                            display(['op ' num2str(opnum) ': test failed']);
                            display('saved tensor');
                            display(test.op);
                            display('calculated tensor');
                            display(op);
                        else
                            display(['op ' num2str(opnum) ': ok']);
                        end
                    end


                    if OUTPUT == 1
                        save([OUTPUT_FOLDER 'op' num2str(opnum)], 'op');
                    end


                    if VERBOSITY==1
                        display(' ');
                        display(' ');
                        display(' ');
                    end

                    opnum = opnum + 1;
                end
            end
        end
    end

end



% helper function to cut parts of complete tensor objects
% example A may have 3 dimensions all full with data
%         but to test [0 3 3] we need to crop the first dimension
function parts = get_tensor_part(tensor, cardinalities, VERBOSITY)
    ndims=length(cardinalities);
    str='tensor(';
    for i=1:ndims
        if cardinalities(i) == 0
            str=strcat(str,'1');
        else
            str=strcat(str,':');
        end
        
        if i ~= ndims
            str=strcat(str,',');
        end
    end
    str=strcat(str,')');
    parts = eval( str );
    if VERBOSITY > 2
        display([str ': '])
        display(parts);
    end
end