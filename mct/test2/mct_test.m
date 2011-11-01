function [] = mct_test()

    % all numbers must be different for proper card_C selection
    V=[2 3 4];

    % all possible tensors cardinality sets using index set V
    cards = perms(V);
    cards_AB = cards;

    % all possible output tensors using cards
    for c=1:size(cards,1)
        card=cards(c,:);
        for choose=1:(length(card)-1)
            %display(['choose ' num2str(choose)])
            % take 'choose' number combinations of card's elements
            choices=nchoosek(card,choose);
            % for each choice row
            for ch=1:size(choices,1)
                %display(['selected choice ' num2str(choices(ch,:))])
                card_tmp = card;
                %generate a combination with selected elements equal to
                %0
                for i=1:size(choices,2)
                    card_tmp(card_tmp==choices(ch,i))=0;
                end
                cards = [cards ; card_tmp];
            end
        end
    end

    cards_C = [cards ; zeros(1, length(V))];

    S = RandStream('mt19937ar');
    RandStream.setDefaultStream(S);

    VERBOSITY=0;

    addpath('../')
    format('compact')
    opnum=0;
    illegal_count=0;
    error=0;

    for card_A_ind=1:size(cards_AB,1)
        for card_B_ind=1:size(cards_AB,1)
            for card_C_ind=1:size(cards_C,1)
                opnum = opnum + 1;

                display(['opnum ' num2str(opnum) '\n\n']);

                card_A = cards_AB(card_A_ind,:);
                card_B = cards_AB(card_B_ind,:);
                card_C = cards_C(card_C_ind,:);

                A=round(rand(card_A)*5);
                B=round(rand(card_B)*5);

                display([ 'card_A ' num2str(card_A) ...
                          ' card_B ' num2str(card_B) ...
                          ' card_C ' num2str(card_C) ])

                if is_legal(card_A, card_B, card_C, VERBOSITY) == 0
                    illegal_count = illegal_count +1;
                    continue
                end

                if VERBOSITY > 0
                    display('input data')
                    display(A)
                    display(B)
                end

                %perform the operation on cpu with matlab code
                tic; [op_m, ~] = tensormul( A, card_A, B, card_B, card_C, ...
                                            VERBOSITY , [] ); time_m = toc;

                display('output_m')
                display(op_m)

                % perform operation on gpu
                tic; op_g=mct('tensor_gpu',A,card_A,B,card_B,card_C,1); ...
                          time_g=toc;

                display('output_g')
                display(op_g)

                % perform operation on cpu with C code
                tic; op_c=mct('tensor_cpp',A,card_A,B,card_B,card_C,1); time_c=toc;

                display('output_c')
                display(op_c)


                display(['timings m ' num2str(time_m) ' g ' ...
                         num2str(time_g) ' c ' num2str(time_c) ...
                        ])


                error = error + test_equality(op_m, op_c, opnum);
                error = error + test_equality(op_c, op_g, opnum);


            end
        end
    end
    display(['illegal_count ' num2str(illegal_count)])
    display(['valid _count ' num2str(opnum-illegal_count)])
    display(['errors ' num2str(error)])
    exit
end


function legal = is_legal(card_A, card_B, card_C, VERBOSITY)

    legal = 1;
    for i=1:length(card_A)
        if ( (card_C(i) ~= card_A(i) || card_C(i) ~= card_B(i)) & card_C(i) ~= 0 ) || ...
            card_A(i) ~= card_B(i)
            display(['cardinality combination not legal on ' ...
                     'dimension ' num2str(i)])
            legal=0;
            return
        end
    end
            
end

function [error] = test_equality(op1, op2, opnum)
    error=0;

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
        error=1;
        display(['op ' num2str(opnum) ': test failed ' ]) % (' num2str(errcount) ')']);
        display('saved tensor');
        display(op2);
        %return
        %errcount=errcount+1;
    else
        display(['op ' num2str(opnum) ': ok']);
    end
end
