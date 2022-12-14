% TGPs for ABGMM features & proposed estimated ground truth
% Note: Add to path following folders: DirectTGP, SMTGP, TGP, thirdparty
clear; clc; close all;
% add minFunc Path
addpath(genpath('./thirdparty/minFunc_2012'));

% load data
load('AMG1608_MoodData.mat');

% X = AMG1608_MoodData.X_FeatStats;
% X = AMG1608_MoodData.X_512_KM;
% X = AMG1608_MoodData.X_512_ABGMM;
% Y = [AMG1608_MoodData.Y_Valence_Avg AMG1608_MoodData.Y_Arousal_Avg];
Y = [AMG1608_MoodData.Y_Valence_CIWM AMG1608_MoodData.Y_Arousal_CIWM];

load('ABGMM_AMG1608_POST_512.mat');
X = AMG1608POST; % non-normalized, i.e. probability values only

N = size(X,1);
% N = 600;

% shuffle X and Y
rp = randperm(N);
X = X(rp,:);
Y = Y(rp,:);

tag = floor(N/2);
TestInput = X(1:tag,:);
Input = X(tag+1:end,:);
TestTarget = Y(1:tag,:);
Target = Y(tag+1:end,:);

fGPR = 0;
fWKNN = 0;
fKLTGP = 0;
fSMTGP = 0;
fDTGP = 1;
% %% Reduced data for debugging/understanding of code
% Input = Input(1:10,641:642);
% TestInput = TestInput(1:10,641:642);
% Target = Target(1:10,1:3);
% TestTarget = TestTarget(1:10,1:3);

%% Gaussian Process Regression
if fGPR
[kparam, lambda] = GPRTuningParams(Input, Target, TestInput, TestTarget, 2);
% kparam = 2;
% lambda = 1e-5;
K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
tic;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
GPTestTime = toc;
[GPError, GPErrorvec] = JointError(GPPred, TestTarget);
disp(['Error of GPR is: ' num2str(GPError)]);

% Obtain covariances: error bars (these are not dependent on y)
tmp = EvalKernel(TestInput,TestInput,'rbf',kparam);
VarGP = diag(tmp) - sum(testK'.*(inv(K+lambda*eye(size(K)))*testK'),1)';
end

%% Weighted K-Nearest Neighbour Regression
if fWKNN
K = WKNNTuningParams(Input, Target, TestInput, TestTarget, 2);
% K = 12;
tic;
WKNNPred = WKNNRegressor(TestInput, Input, Target, K);
WKNNTestTime = toc;
[WKNNError, WKNNErrorvec] = JointError(WKNNPred, TestTarget);
disp(['Error of WKNN is: ' num2str(WKNNError)]);
end

%% KL TGP
if fKLTGP
% [kparam1, kparam2] = KLTGPTuningParams(Input, Target, TestInput, TestTarget, 2);
% KLTGPParam.kparam1 = kparam1; % 0.2;
% KLTGPParam.kparam2 = kparam2; % 20;
% 
KLTGPParam.kparam1 = 0.2;
KLTGPParam.kparam2 = 1e-5;%2*1e-6;
KLTGPParam.lambda = 1e-3;
KLTGPParam.knn = 100;
% Param.SMAlpha = 0.5 ;
% Param.SMBeta = 0.5;

[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, KLTGPParam);
tic;
TGPPredKL = TGPTest(TestInput, Input, Target, KLTGPParam, InvIK, InvOK);
TGPTestTime = toc;
[KLTGPError, KLTGPErrorvec] = JointError(TGPPredKL, TestTarget);
disp(['Error of KLTGP is: ' num2str(KLTGPError)]);

% % KLTGP with KNN (Twin Gaussian Processes with K Nearest Neighbors)
% Very time-consuming, since InvIK and InvOK computed for each input based
% on nearest neigbors!
% tic;
% KLTGPKNNPred = TGPKNN(TestInput, Input, Target, KLTGPParam);
% KLTGPKNNTestTime = toc;
% [KLTGPKNNError, KLTGPKNNErrorvec] = JointError(KLTGPKNNPred, TestTarget);
% disp(['Error of KLTGPKNN: ' num2str(KLTGPKNNError)]);
end

%% Sharma Mittal (TGP SH)
if fSMTGP
load('TGPPredKL.mat');
% [kparam1, kparam2, alpha, beta] = SMTGPTuningParams(Input, Target, TestInput, TestTarget, TGPPredKL, 2);
SMTGPParam.kparam1 = 0.2;
SMTGPParam.kparam2 = 2*1e-6;
SMTGPParam.lambda = 1e-3;
% Values alpha and beta could take (for cross validation
% alphas = [0.00000001, 0.000001,0.00001, 0.1, 0.2, 0.29, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999];
% betas = [ 7, 5, 3.5, 2.5, 2, 1.5, 0.999];
% betas = [ 1.5,2, 2.5,3,5,7];

SMTGPParam.SMAlpha =  0.4;
SMTGPParam.SMBeta  =  0.99;
[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, SMTGPParam);
IOKAlphaInv =  inv((1-SMTGPParam.SMAlpha)* IK+ (SMTGPParam.SMAlpha)*OK);
% Note: SMTGP uses TGPPredKL as initial targets, so count time of KLTGP test time also!
tic;
[TGPPredSM, TGPcertSH] = TGPSH4Test(TestInput, Input, Target, SMTGPParam, InvIK, InvOK, IK, OK, TGPPredKL, IOKAlphaInv);
SMTGPTestTime = toc;
[SMTGPError, SMTGPErrorvec] = JointError(TGPPredSM, TestTarget);
certainties = mean(TGPcertSH);
disp(['Error of SMTGP is: ' num2str(SMTGPError),', certainties of SMTGP is: ', num2str( mean(TGPcertSH)),', certainties std of SMTGP is: ' num2str( std(TGPcertSH))]);
end

%% Direct Importance Weighted TGP (DIWTGP)
if fDTGP
% Determine Importance Weights
wh_xtr = ones(1,size(Input,1)); % With this option, DIWTGP = DTGP only
% alphaparam = 0.5;
% [PE1,wh_xtr,wh_xte]=RuLSIF(TestInput',Input',TestInput',alphaparam,[],[],[],5);

[kparam1, kparam2] = DTGPTuningParamsR2(Input, Target, TestInput, TestTarget, wh_xtr);
DIWTGP_Param.kparam1 = kparam1;   
DIWTGP_Param.kparam2 = kparam2;

% DIWTGP_Param.kparam1 = 1e-5;%0.2;
% DIWTGP_Param.kparam2 = 1e-1;%2*1e-6;
% DIWTGP_Param.kparam1 = 0.15;%0.2;   
% DIWTGP_Param.kparam2 = 1e-5;%2*1e-6;
% DIWTGP_Param.kparam3 = DIWTGP_Param.kparam2;
DIWTGP_Param.lambda = 10^(-3);
DIWTGP_Param.knn1 = min(size(Input,1),500); % M nearest neighbor
DIWTGP_Param.wknnflag = 1; % Distance based weighting
DIWTGP_Param.knn2 = min(size(Input,1)/2,100); % K nearest neighbor

%Direct IWTGP
[DIWTGPKNNPred traintime testtime mu_all] = DWTGPKNN(TestInput, Input, Target,DIWTGP_Param,wh_xtr);
[R2_DTGP, MSE_DTGP] = JointR2(DIWTGPKNNPred, TestTarget);
disp(['R2 of DIWTGP is: ' num2str(R2_DTGP)]);
end

%% Test times of TGPs
% disp(['Test time of GPR is: ' num2str(GPTestTime)]);
% disp(['Test time of WKNN is: ' num2str(WKNNTestTime)]);
% disp(['Test time of KLTGP is: ' num2str(TGPTestTime)]);
% % disp(['Test time of KLTGPKNN is: ' num2str(KLTGPKNNTestTime)]);
% disp(['Test time of SMTGP is: ' num2str(SMTGPTestTime)]);
% disp(['Test time of DIWTGP is: ' num2str(testtime)]);

%% Hilbert-Schmidt Independent Criterion with K Nearest Neighbors
% Param.knn = 100;
% Param.kparam1 = 100;
% Param.kparam2 = 2*1e-5;
% Param.kparam1 =  2*1e-4;
% Param.kparam2 = 2*1e-4;
% HSICKNNPred = HSICKNN(TestInput, Input, Target, Param);
% %[HSICKNNError, HSICKNNErrorvec] = JointError(HSICKNNPred, TestTarget,1);
% disp(['Error of HSICKNN is: ' num2str(mean(abs(HSICKNNPred(:)-TestTarget(:))))]);
% %disp(['HSICKNN: ' num2str(HSICKNNError)]);
% 
%% Kernel Target Alignment with K Nearest Neighbors
% Param.kparam1 =  2*1e-3;
% Param.kparam2 = 2*1e-3;
% KTAKNNPred = KTAKNN(TestInput, Input, Target, Param);
% %[KTAKNNError, KTAKNNErrorvec] = JointError(KTAKNNPred, TestTarget);
% %disp(['KTAKNN: ' num2str(KTAKNNError)]);
% disp(['Error of KTAKNN is: ' num2str(mean(abs(KTAKNNPred(:)-TestTarget(:))))]);
% 
