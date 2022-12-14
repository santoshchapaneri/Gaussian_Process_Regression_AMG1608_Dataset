function [kparam, lambda] = GPRTuningParams(Input, Target, TestInput, TestTarget, nDataset)

% Gaussian Process Regression 
% Cross validation to find optimal hyperparameters
kparamset = [1e-5 1e-2 0.5 1 2 5 10 20 25];
lambdaset = [1e-7 1e-5 1e-3 0.5 1];
[F, S] = ndgrid(kparamset, lambdaset);
% Run a fitting on every pair fittingfunction(F(J,K), S(J,K))
fitresult = arrayfun(@(p1,p2) myTuningGPR(p1,p2, Input, Target, TestInput, TestTarget, nDataset), F, S); 
[v1, i1] = min(fitresult);
[~, i2] = min(v1);
kparam = kparamset( i1(i2) );
lambda = lambdaset(i2);
end

function GPError = myTuningGPR(kparam, lambda, Input, Target, TestInput, TestTarget, nDataset)

K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
switch nDataset
    case 1
        GPError =  mean(abs(GPPred(:)-TestTarget(:)));
    case 2
        [GPError, ~] = JointError(GPPred, TestTarget);
end
end