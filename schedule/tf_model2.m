
% tucker_model

dim_i=TFDimension;
dim_i.name='i';
dim_i.cardinality=5;

dim_j=TFDimension;
dim_j.name='j';
dim_j.cardinality=6;

dim_k=TFDimension;
dim_k.name='k';
dim_k.cardinality=7;

dim_p=TFDimension;
dim_p.name='p';
dim_p.cardinality=8;

dim_q=TFDimension;
dim_q.name='q';
dim_q.cardinality=9;

dim_r=TFDimension;
dim_r.name='r';
dim_r.cardinality=10;



A=TFFactor;
A.name='A';
A.isLatent=1;
A.dims=[dim_i dim_p];

B=TFFactor;
B.name='B';
B.isLatent=1;
B.dims=[dim_j dim_q];

C=TFFactor;
C.name='C';
C.isLatent=1;
C.isInput=1;
C.dims=[dim_k dim_r];

G=TFFactor;
G.name='G';
G.isLatent=1;
G.dims=[dim_p dim_q dim_r];

X=TFFactor;
X.name='X';
X.isObserved=1;
X.dims=[dim_i dim_j dim_k];


tucker_model = TFModel;
tucker_model.name = 'Tucker3';
tucker_model.factors = [A B C G X];
tucker_model.dims=[dim_i dim_j dim_k dim_p dim_q dim_r];


all_dims=tucker_model.dims;
A.rand_init(all_dims, 100);
B.rand_init(all_dims, 100);
C.rand_init(all_dims, 100);
G.rand_init(all_dims, 100);
X.rand_init(all_dims, 100);



% parafac model

p_A=TFFactor;
p_A.name='p_A';
p_A.isLatent=1;
p_A.dims=[dim_i dim_r];
p_A.rand_init(all_dims, 100);

p_B=TFFactor;
p_B.name='p_B';
p_B.isLatent=1;
p_B.dims=[dim_j dim_r];
p_B.rand_init(all_dims, 100);

p_C=TFFactor;
p_C.name='p_C';
p_C.isLatent=1;
p_C.dims=[dim_k dim_r];
p_C.rand_init(all_dims, 100);

parafac_model = TFModel;
parafac_model.name = 'Parafac';
parafac_model.factors = [p_A p_B p_C X];
parafac_model.dims=[dim_i dim_j dim_k dim_r];




% GCTF model from tucker + parafac
gctfmodel=GCTFModel;
gctfmodel.tfmodels=[tucker_model parafac_model];



% VISUALIZE

if exist('VISUALIZE')
    [dn fn edges] = tucker_model.print();
    system(['python fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

    pause

    [dn fn edges] = parafac_model.print();
    system(['python fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

    pause

    % visualize GCTF model
    [dn fn edges] = gctfmodel.print();
    system(['python fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

end


