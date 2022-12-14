function [kparam1, kparam2, alpha, beta] = SMTGPTuningParams(Input, Target, TestInput, TestTarget, TGPPredKL, nDataset)

% SM Twin Gaussian Process Regression 
% Cross validation to find optimal hyperparameters
% kparam1set = [1e-5 1e-2 0.5 1 2 5 10 20 25];
% kparam2set = [1e-5 1e-2 0.5 1 2 5 10 20 25];
kparam1set = 10;
kparam2set = 1e-5;
% SMAlphaset = [1e-7, 1e-5, 1e-3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9999];
SMAlphaset = [0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99];
% SMBetaset = [7, 5, 3.5, 2.5, 2, 1.5, 0.999];
SMBetaset = [0.99, 0.5, 1.5];
[F, S, A, B] = ndgrid(kparam1set, kparam2set, SMAlphaset, SMBetaset);
% [A, B] = ndgrid(SMAlphaset, SMBetaset);
% Run a fitting on every pair fittingfunction(F(J,K), S(J,K))
fitresult = arrayfun(@(p1, p2, p3, p4) myTuningSMTGP(p1, p2, p3, p4, Input, Target, TestInput, TestTarget, TGPPredKL, nDataset), F, S, A, B); 
[~, minidx] = min(fitresult(:));
[i, j, k, l] = ind2sub( size(fitresult), minidx );
kparam1 = kparam1set(i);
kparam2 = kparam2set(j);
alpha = SMAlphaset(k);
beta = SMBetaset(l);
end

function SMTGPError = myTuningSMTGP(kparam1, kparam2, alpha, beta, Input, Target, TestInput, TestTarget, TGPPredKL, nDataset)
SMTGPParam.kparam1 = kparam1; % kparams are from KLTGP tuning
SMTGPParam.kparam2 = kparam2;
SMTGPParam.lambda = 1e-4;
SMTGPParam.SMAlpha = alpha;
SMTGPParam.SMBeta  =  beta;
[InvIK, InvOK, IK, OK] = TGPTrain(Input, Target, SMTGPParam);
IOKAlphaInv =  inv((1-SMTGPParam.SMAlpha)* IK+ (SMTGPParam.SMAlpha)*OK);
[TGPPredSM, ~] = TGPSH4Test(TestInput, Input, Target, SMTGPParam, InvIK, InvOK, IK, OK, TGPPredKL, IOKAlphaInv);
switch nDataset
    case 1
        SMTGPError = mean(abs(TGPPredSM(:)-TestTarget(:)));
    case 2
        [SMTGPError, ~] = JointError(TGPPredSM, TestTarget);
end
fprintf('k1:%f, k2:%f, a:%f, b:%f, Error:%f\n', kparam1, kparam2, alpha, beta, SMTGPError); % for debugging only
end