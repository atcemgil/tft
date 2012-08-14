% Examples for Tensor Factorization Toolbox
% for details see index.html page

% tucker_model

dim_i=TFDimension('name', 'i', 'cardinality', 5);
dim_j=TFDimension('cardinality', 6, 'name', 'j');
dim_k=TFDimension('cardinality', 7, 'name', 'k');
dim_p=TFDimension('cardinality', 8, 'name', 'p');
dim_q=TFDimension('cardinality', 9, 'name', 'q');
dim_r=TFDimension('cardinality', 10, 'name', 'r');

A=TFFactor('name', 'A', 'type', 'latent', 'dims', [dim_i dim_p]);
B=TFFactor('name', 'B', 'type', 'latent', 'dims', [dim_j dim_q]);
C=TFFactor('name', 'C', 'type', 'latent', 'dims', [dim_k dim_r], ...
           'isClamped', true);
G=TFFactor('name', 'G', 'type', 'latent', 'dims', ...
           [dim_p, dim_q, dim_r]);
X=TFFactor('name', 'X', 'type', 'observed', 'dims', ...
           [dim_i, dim_j, dim_k]);

tucker_model = TFModel('name', 'Tucker3', 'factors', [A B C G X], ...
                       'dims', [dim_i dim_j dim_k dim_p dim_q dim_r]);

tucker_model.rand_init_latent_factors('nonClamped');
X.rand_init(tucker_model.dims, 100) % init observation
C.rand_init(tucker_model.dims, 100) % init clamped


% parafac model

p_A=TFFactor('name', 'p_A', 'type', 'latent', 'dims', [dim_i dim_r]);
p_B=TFFactor('name', 'p_B', 'type', 'latent', 'dims', [dim_j dim_r]);
p_C=TFFactor('name', 'p_C', 'type', 'latent', 'dims', [dim_k dim_r]);

parafac_model = TFModel('name', 'Parafac', ...
                        'factors', [p_A p_B p_C X], ...
                        'dims', [dim_i dim_j dim_k dim_r]);

parafac_model.rand_init_latent_factors('all');





% GCTF model from tucker + parafac
%gctfmodel=GCTFModel;
%gctfmodel.tfmodels=[tucker_model parafac_model];



% VISUALIZE

if exist('VISUALIZE_UBIGRAPH')
    [dn fn edges] = tucker_model.print_ubigraph();
    system(['python visualize/fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

    pause

    [dn fn edges] = parafac_model.print();
    system(['python visualize/fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

    %pause

    % visualize GCTF model
    %[dn fn edges] = gctfmodel.print();
    %system(['python fgplot.py "' dn '" "' fn '" "' edges '"'  ]);

end


if exist('VISUALIZE_DOT')
    g = tucker_model.schedule_dp();
    system([ 'rm /tmp/img.eps; echo '' ' g.print_dot  [' '' |' ...
                        ' dot -o /tmp/img.eps ;  display  /tmp/img.eps; ' ] ] );
end

if exist('PROFILE_PLTF')
    N=10;
    interval=20;
    times=zeros(1,length(1:interval:N*interval));
    j=1;
    for i=1:interval:N*interval
        j
        rng(1)
        dim_p=TFDimension('cardinality', 8+i, 'name', 'p');
        dim_q=TFDimension('cardinality', 9+i, 'name', 'q');
        dim_r=TFDimension('cardinality', 10+i, 'name', 'r');

        A=TFFactor('name', 'A', 'type', 'latent', 'dims', [dim_i dim_p]);
        B=TFFactor('name', 'B', 'type', 'latent', 'dims', [dim_j dim_q]);
        C=TFFactor('name', 'C', 'type', 'latent', 'dims', [dim_k dim_r], ...
                   'isClamped', true);
        G=TFFactor('name', 'G', 'type', 'latent', 'dims', ...
                   [dim_p, dim_q, dim_r]);
        X=TFFactor('name', 'X', 'type', 'observed', 'dims', ...
                   [dim_i, dim_j, dim_k]);

        tucker_model = TFModel('name', 'Tucker3', 'factors', [A B C G X], ...
                               'dims', [dim_i dim_j dim_k dim_p dim_q dim_r]);

        tucker_model.rand_init_latent_factors('nonClamped');
        X.rand_init(tucker_model.dims, 100) % init observation
        C.rand_init(tucker_model.dims, 100) % init clamped


        tic; tucker_model.pltf(30); times(j)=toc;
        j = j+1;
    end
    plot(1:length(1:interval:N*interval), times);
    title('Tucker3 Model - Increment pqr Dimensions');
    xlabel('Increments');
    ylabel('Seconds');

end