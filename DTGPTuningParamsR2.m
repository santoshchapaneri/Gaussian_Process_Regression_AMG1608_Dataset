function [kparam1, kparam2] = DTGPTuningParamsR2(Input, Target, TestInput, TestTarget, wh_xtr)

% SM Twin Gaussian Process Regression 
% Cross validation to find optimal hyperparameters

% kparam1set = [1e-5];
% kparam2set = [1e-5];

% kparam1set = [1e-5 1e-2 0.5 1 2 5 10 20 25];
% kparam2set = [1e-5 1e-2 0.5 1 2 5 10 20 25];
% kparam1set = [0.5 1 2 5 10 20 25 50]; % this is 2 * sigmax^2
kparam1set = [1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1 2 5 10 25 50 100 1000 5000 10000 1e+5]; % this is 2 * sigmax^2
% kparam1set = [1e-6 1e-5 1e-4 1e-3 1e-2 0.5 1 2 5 10]; % this is 2 * sigmax^2
% kparam1set = 1./kparam1set;
kparam2set = [1e-6 1e-5 1e-4 1e-3 1e-2 0.1 1 2 5 10 25 50 100 1000 5000 10000 1e+5]; % this is 2 * sigmay^2
% kparam2set = [1e-6 1e-5 1e-4 1e-3 1e-2 0.5 1 2 5 10]; % this is 2 * sigmay^2
% kparam2set = 1./kparam2set;
% kparam1set = 10;
% kparam2set = 25;
[F, S] = ndgrid(kparam1set, kparam2set);
% Run a fitting on every pair fittingfunction(F(J,K), S(J,K))
[DTGP_R2] = arrayfun(@(p1, p2) myTuningDTGP(p1, p2, Input, Target, TestInput, TestTarget, wh_xtr), F, S); 
[~, minidx] = min(DTGP_R2(:));
[i, j] = ind2sub( size(DTGP_R2), minidx );
kparam1 = kparam1set(i);
kparam2 = kparam2set(j);
end

function [DTGP_R2] = myTuningDTGP(kparam1, kparam2, Input, Target, TestInput, TestTarget, wh_xtr)
DIWTGP_Param.kparam1 = kparam1;
DIWTGP_Param.kparam2 = kparam2;
DIWTGP_Param.lambda = 10^(-3);
% DIWTGP_Param.knn1 = min(length(Input),200); % M nearest neighbor
DIWTGP_Param.knn1 = min(length(Input),500); % M nearest neighbor
DIWTGP_Param.wknnflag = 1; % Distance based weighting
% DIWTGP_Param.knn2 = 25; % K nearest neighbor
DIWTGP_Param.knn2 = 100; % K nearest neighbor

%Direct IWTGP
[DIWTGPKNNPred, ~, ~, ~] = DWTGPKNN(TestInput, Input, Target,DIWTGP_Param,wh_xtr);
[DTGP_R2, ~] = JointR2(DIWTGPKNNPred, TestTarget);
DTGP_R2 = DTGP_R2(2);
fprintf('k1:%f, k2:%f, R2:%f\n', kparam1, kparam2, DTGP_R2); % for debugging only
end