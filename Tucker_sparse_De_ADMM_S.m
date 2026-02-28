function [Y]= Tucker_sparse_De_ADMM_S( oriData3_noise, data_HSI, data_MS, Par)

%% ≥ı ºªØ
epsilon   = 0.00001;
normD     = norm(oriData3_noise(:)); 
[C1, U1]  = tensorSVD(oriData3_noise,Par.rank1);
Q1        = zeros(size(C1));
Lam1      = Q1;
[c1_m,c1_n,c1_z] = size(C1);
sizeC1           = size(C1);
[C2, U2]  = tensorSVD(data_MS,Par.rank2);
Q2        = zeros(size(C2));
Lam2      = Q2;

[c2_m,c2_n,c2_z] = size(C2);
sizeC2           = size(C2);
C1_matrix = reshape(C1,c1_m*c1_n,c1_z);
C2_matrix = reshape(C2,c2_m*c2_n,c2_z);
P         = C2_matrix' * pinv(C1_matrix'); 

Y         = oriData3_noise;
E_Y       = Y;
sizeY     = size(Y);
S         = zeros(sizeY);
ndim      = length(sizeY);
X         = data_MS;
sizeX     = size(X);
E_X       = X;
%% update U1
for iter  = 1:Par.Iter
    for j = 1:ndim
        unfoTemp    = Unfold(oriData3_noise-S, sizeY, j);
        sizeC       = Par.rank1;
        tempC       = C1;
        for k1 = [1:j-1,j+1:ndim]
            tempC     = ttm1D(tempC,U1{k1},k1,sizeC,sizeY(k1));
            sizeC(k1) = sizeY(k1);
        end
        UnfoldC     = Unfold( tempC, Par.rank1, j);
        tempMatix   = unfoTemp*UnfoldC';
        [V1,~,V2]   = svd(tempMatix,'econ');
        U1{j}       = V1*V2';
    end
    


%% update Q1
Q1        = softthre(C1+Lam1/Par.mu,Par.lambda1/Par.mu );
Q1_matrix = reshape(Q1,c1_m*c1_n,c1_z);

%% update C1
TempC1    = reshape((P' * C2_matrix')',c1_m,c1_n,c1_z);
TempC3    = Par.mu * Q1- Lam1;
TempC2    = oriData3_noise - S;
sizeCY    = sizeY;
    for k = 1:ndim 
TempC2    = ttm1D(TempC2,U1{k}',k,sizeCY,Par.rank1(k));
sizeCY(k) = Par.rank1(k);
    end

TempC1_sp = (Par.beta*(P')*P + (Par.mu +1)*eye(size((P')*P)));   
TempC1_matrix    = reshape(Par.beta * TempC1 + TempC2 + TempC3, c1_m*c1_n,c1_z);
C1_matrix        = (pinv(TempC1_sp)*  TempC1_matrix')';
C1               = reshape(C1_matrix, c1_m,c1_n,c1_z);

%% calculating Y   
    preY    = Y;
    sizeD   = Par.rank1;
    Y1       = C1;
    for ky  = 1:ndim
        Y1   = ttm1D(Y1,U1{ky},ky,sizeD,sizeY(ky));
        sizeD(ky) = sizeY(ky);
    end
Y=Y1;
%% update U2

    for j2 = 1:ndim
        unfoTemp2    = Unfold(X, sizeX, j2);
        sizeC2       = Par.rank2;
        tempu2       = C2;
        for k2 = [1:j2-1,j2+1:ndim]
            tempu2    = ttm1D(tempu2,U2{k2},k2,sizeC2,sizeX(k2));
            sizeC2(k2) = sizeX(k2);
        end
        UnfoldC2     = Unfold( tempu2, Par.rank2, j2);
        tempMatix2   = unfoTemp2*UnfoldC2';
        [V12,~,V22]   = svd(tempMatix2,'econ');
        U2{j2}     = V12*V22';
    end


%% update Q2
Q2        = softthre(C2+Lam2/Par.mu,Par.lambda2/Par.mu );
Q2_matrix = reshape(Q2,c2_m*c2_n,c2_z);

%% update C2
TempC21     = reshape(Par.beta * (P * C1_matrix')',c2_m,c2_n,c2_z);
TempC23     = Par.mu * Q2 - Lam2;
TempC22     = X;
sizeCX      = sizeX;
for kc2 = 1:ndim 
TempC22     = ttm1D(TempC22,U2{kc2}',kc2,sizeCX,Par.rank2(kc2));
sizeCX(kc2) = Par.rank2(kc2);
end

C2          = (TempC21 + TempC22 + TempC23)/(Par.beta + Par.mu + 1);
C2_matrix   = reshape(C2,c2_m*c2_n,c2_z);
      
%% update S 
S          = softthre(oriData3_noise - Y,Par.lambda);

%% update P

Temp4 = Par.beta * C2_matrix' * C1_matrix;
P     = Temp4 * pinv(Par.gamma*eye(size( C1_matrix'* C1_matrix)) + Par.beta * (C1_matrix'* C1_matrix));

%% update M
Lam1     = Lam1 + Par.mu*(C1 - Q1);
Lam2     = Lam2 + Par.mu*(C2 - Q2);
Par.mu   = Par.mu*1.5;

errList(iter)    = norm(Y(:)-preY(:)) / normD;
%     fprintf('JD: iterations = %d   difference=%f\n', iter, errList(iter));
    if errList(iter) < epsilon
        break;  
    end 
    
% [PSNR,SSIM,~] = evaluate(data_HSI,Y,size(data_HSI,1),size(data_HSI,2));
%  %      [PSNR,SSIM,~,~] = evaluate(O_Img,E_Img,Height,Width);
%     PSNR = mean(PSNR);SSIM = mean(SSIM);
% %    PSNR = mean(0);SSIM = mean(0);
%     fprintf( 'Iter = %2.3f, PSNR = %2.2f, SSIM = %2.3f \n', iter, PSNR, SSIM);
   

 
 end



end