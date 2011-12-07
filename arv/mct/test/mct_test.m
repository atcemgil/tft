
% test_card_num: number of cardinalities to test
% example: input of number 3 indicates conducting two different
%          tests with 2 and 3 dimensions.

% max_card: maximum cardinality for test cardinality dimensions

% example: given call mct_test(4, 3) we have the following tests:
% A * B = C
% A and B can be one of the following tensors
% 1. [1 1]
% 2. [1 2]
% 3. [2 1]
% 4. [1 3]
% 5. [2 2]
% 6. [3 1]
% 7. [3 2]
% 8. [2 3]
% 9. [3 3]

% 10. [2 2 2]
% 11. [2 2 3]
% ...

% 37. [1 1 1 1]
% 38. [1 1 1 2]
% ...
% 117. [ 3 3 3 3]

% A and B would have any one of 117 possible cardinalities,
% making a total of 117^2 = 13689 possible different combinations

function [] = mct_test(max_test_card_num, max_card)
    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);

    VERBOSITY=3;

    addpath('../')
    format('compact')
    opnum=1;

    for test_card_num=2:max_test_card_num
        cards = gen_combination(test_card_num, max_card);
        for card_A_ind=1:size(cards,1)
            for card_B_ind=1:size(cards,1)
                for card_C_ind=1:size(cards,1)

                    display(['opnum ' num2str(opnum)]);

                    card_A = cards(card_A_ind,:);
                    card_B = cards(card_B_ind,:);
                    card_C = bsxfun(@minus, cards(card_C_ind,:), 1);

                    if VERBOSITY > 0
                            display([ 'card_A ' num2str(card_A) ...
                                      ' card_B ' num2str(card_B) ...
                                      ' card_C ' num2str(card_C) ])
                    end

                    A=round(rand(card_A)*5);
                    B=round(rand(card_B)*5);

                    display('input data')
                    display(A)
                    display(B)
                    

                    % perform the operation on cpu with matlab code
                    tic; [op_m, ~] = ...
                        tensormul( A, card_A, B, card_B, card_C, ...
                                   VERBOSITY , [] ); time_m = toc;

                    display('output')
                    display(op_m)

                    % perform operation on gpu
                    %tic; op_g=mct('tensor_gpu',A,card_A,B,card_B,card_C,1); time_g=toc;

                    % perform operation on cpu with C code
                    %tic; op_c=mct('tensor_cpp',A,card_A,B,card_B,card_C,1); time_c=toc;

                    %display(['timings m ' num2str(time_m) ' g ' ...
                    %         num2str(time_g) ' c ' num2str(time_c) ...
                    %        ])

                    %test_equality(op_m, op_g, opnum);
                    %test_equality(op_g, op_c, opnum);
                    opnum = opnum + 1;
                end
            end
        end
    end

    exit
end

function val = output_larger_input(card_A, card_B, card_C)
    for i=1:numel(card_A)
        if card_C(i) > card_A(i) || card_C(i) > card_B(i)
            val=1
            return
        end
    end
    val = 0
end            
        

function [] = test_equality(op1, op2, opnum)
    numeldiff = numel(op1) - numel(op2);
    allequal=0;
    if numeldiff == 0
        display(['numeldiff ok : ' num2str(numeldiff)]);
        for n=1:numel(op1)
            if op1(n) ~= op2(n)
                allequal = allequal+1;
            end
        end
    end

    if allequal ~= 0 || numeldiff ~= 0
        display(['op ' num2str(opnum) ': test failed ' ]) % (' num2str(errcount) ')']);
        display('saved tensor');
        display(op2);
        %return
        %errcount=errcount+1;
    else
        display(['op ' num2str(opnum) ': ok']);
    end
end



function [cards card] = gen_combination(card_num, max_card, cards, ...
                                        card)
    if nargin == 2
        card = [];
        cards = [];
    end
    if length(card) == card_num
        cards = [cards ; card];
        return
    else
        for mc = 1:max_card
            [cards, ~] = gen_combination(card_num, max_card, cards, ...
                                         [card mc]);
        end
    end
end