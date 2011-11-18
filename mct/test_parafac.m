
function [] = test_parafac(A,B,C,I,J,K,a,X_true,txt)
  X = get_parafac(A,B,C,I,J,K,a,size(X_true));

  display('selm');
  if sum(sum(sum(X_true==X))) ~= numel(X_true)
    display([ txt ' ERROR'])
  else
    display([txt ' ok'])
  end
end
