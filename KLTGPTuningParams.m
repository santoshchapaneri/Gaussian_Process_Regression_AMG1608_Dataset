function [kparam1, kparam2] = KLTGPTuningParams(Input, Target, TestInput, TestTarget, nDataset)

% KL Twin Gaussian Process Regression 
% Cross validation to find optimal hyperparameters
% kparam1set = [1e-5 1e-2 0.5 1 2 5 10 20 25];
% kparam2set = [1e-5 1e-2 0.5 1 2 5 10 20 25];

kparam1set = [0.5 1 2 5 10 20 25 50]; % this is 2 * sigmax^2
kparam1set = 1./kparam1set;
kparam2set = [0.05 0.5 1 2 5 10 25 50 100 200 500 1000 5000 10000 50000]; % this is 2 * sigmay^2
kparam2set = 1./kparam2set;

[F, S] = ndgrid(kparam1set, kparam1set);
% Run a fitting on every pair fittingfunction(F(J,K), S(J,K))
fitresult = arrayfun(@(p1,p2) myTuningKLTGP(p1, p2, Input, Target, TestInput, TestTarget, nDataset), F, S); 
[v1, i1] = min(fitresult);
[~, i2] = min(v1);
kparam1 = kparam1set( i1(i2) );
kparam2 = kparam2set(i2);
end

function KLTGPError = myTuningKLTGP(kparam1, kparam2, Input, Target, TestInput, TestTarget, nDataset)
KLTGPParam.kparam1 = kparam1;
KLTGPParam.kparam2 = kparam2;
KLTGPParam.kparam3 = KLTGPParam.kparam2;
KLTGPParam.lambda = 1e-4;
[InvIK, InvOK, ~, ~] = TGPTrain(Input, Target, KLTGPParam);
TGPPredKL = TGPTest(TestInput, Input, Target, KLTGPParam, InvIK, InvOK);
switch nDataset
    case 1
        KLTGPError = mean(abs(TGPPredKL(:)-TestTarget(:)));
    case 2
        [KLTGPError, ~] = JointError(TGPPredKL, TestTarget);
end
fprintf('k1: %d, k2: %d, Error: %f\n', kparam1, kparam2, KLTGPError); % for debugging only
end