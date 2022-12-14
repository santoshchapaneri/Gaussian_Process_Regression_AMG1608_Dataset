function [kparam1, kparam2] = DTGPTuningParams(Input, Target, TestInput, TestTarget, wh_xtr, nDataset)

% SM Twin Gaussian Process Regression 
% Cross validation to find optimal hyperparameters
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
fitresult = arrayfun(@(p1, p2) myTuningDTGP(p1, p2, Input, Target, TestInput, TestTarget, wh_xtr, nDataset), F, S); 
[~, minidx] = min(fitresult(:));
[i, j] = ind2sub( size(fitresult), minidx );
kparam1 = kparam1set(i);
kparam2 = kparam2set(j);
end

function DTGPError = myTuningDTGP(kparam1, kparam2, Input, Target, TestInput, TestTarget, wh_xtr, nDataset)
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
switch nDataset
    case 1
        DTGPError = mean((TestTarget(:) - DIWTGPKNNPred).^2);
    case 2
        [DTGPError, ~] = JointError(DIWTGPKNNPred, TestTarget);
end
fprintf('k1:%f, k2:%f, Error:%f\n', kparam1, kparam2, DTGPError); % for debugging only
end