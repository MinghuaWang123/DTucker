  function   Y = ttm1D (X,U,k,sizeX,Um)
Y = shiftdim( reshape(U*reshape(  shiftdim(X,k-1) , sizeX(k),[] ), [Um,sizeX(k+1:3),sizeX(1:k-1)]), 3+1-k);
end