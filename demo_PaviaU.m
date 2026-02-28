close all;
clear all;
clc;

load('D:\PAV01.mat')

[M N p] = size(data_HSI);
%% Pavia case 4
Par.rank1   = [130,130,3];
Par.rank2   = [130,130,3];
Par.beta    = 3;
Par.gamma   = 1;
Par.lambda  = 0.06;
Par.Iter    = 30;

Par.lambda1 = 0.03;
Par.lambda2 = 0.03;

Par.mu      = 0.001;

tic
[output_imageTucker]= Tucker_sparse_De_ADMM_S( oriData3_noise, data_HSI, data_MSI, Par);    
time_Joint_DeNoising = toc;
Joint_DeNoising = calcDenoiseResult( data_HSI,oriData3_noise,output_imageTucker,'Jointdenoising',false );
ergas_Tucker_De = ErrRelGlobAdimSyn(data_HSI,output_imageTucker);

A       = reshape(data_HSI,M*N,p);
B       = reshape(output_imageTucker,M*N,p);
msad_JDTucker = mSAD(A,B);

