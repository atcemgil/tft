
x1_card_sym = ['i','j']
x1=rand(3,4)
x2_card_sym = ['j','k']
x2=rand(4,5)

z1_card_sym = ['i','j','k']
z1=rand(3,4,5)

z2_card_sym = ['i','j']
z2=rand(3,4)

z3_card_sym = ['j','k']
z3=rand(4,5)

gctf_seq(5, ['i','j','k'], [3,4,5] , 1, zeros(2,3), ...
         x1_card_sym, x1, ...
         x2_card_sym, x2, ...
         z1_card_sym, z1, 0 ,...
         z2_card_sym, z2, 1 ,...
         z3_card_sym, z3, 0 )
