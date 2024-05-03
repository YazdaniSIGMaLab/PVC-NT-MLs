clear all
close all
clc

% data = importdata('Input-output-ML-C3N.xlsx');
% 
% data = data.data;
% 
% Input_NN = data(:,1:end-1);
% Output_NN = data(:,end);

load('trainedModel_GPR_CNT')
load('trainedModel_GPR_C3N')
load('trainedModel_SVM_CNT')
load('trainedModel_SVM_C3N')

N = 7; %N = 1 , 3, 7, 11 
alfa = 0.07; %[0, 0.07, 0.12, 0.23]
Ori = 3; %X = 1, Y = 2, Z = 3
n = 21;

N = ones(n,1)*N;
alfa = ones(n,1)*alfa;
Ori = ones(n,1)*Ori;
strain = linspace(0,0.05,n)';

Input_pred = [N alfa Ori strain];

%y_NN = myNeuralNetworkFunction_CNT(Input_pred);
y_NN = myNeuralNetworkFunction_C3N(Input_pred);

%y_GPR = trainedModel_GPR_CNT.predictFcn([Input_pred]);
y_GPR = trainedModel_GPR_C3N.predictFcn([Input_pred]);

%y_SVM = trainedModel_SVM_CNT.predictFcn([Input_pred]);
y_SVM = trainedModel_SVM_C3N.predictFcn([Input_pred]);

plot(strain, y_NN, '-^b')
hold on
plot(strain, y_GPR, '-ok')
hold on
plot(strain, y_SVM, '-*r')





