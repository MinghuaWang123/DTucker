close all;
clc;
clear all;

load('D:\GF5_6_airport.mat');

tic
rankk = 270; 
Par.rank1   = [rankk,rankk,10]; % rank2 = 2
Par.rank2   = [rankk,rankk,3];
Par.beta    = 0.5;
Par.gamma   = 1;
Par.lambda  = 0.8;
Par.Iter    = 30;
Par.lambda1 = 0.03;
Par.lambda2 = 10;
Par.mu      = 0.01;

[output_image_JD]= Tucker_sparse_De_ADMM_S_GF( data_HSI, data_MS, Par); 
Joint_DeNoising_relax_Indian_TIME = toc;




