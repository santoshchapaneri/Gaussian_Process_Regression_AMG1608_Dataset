function K = WKNNTuningParams(Input, Target, TestInput, TestTarget, nDataset)

% Gaussian Process Regression 
% Cross validation to find optimal hyperparameters
Kset = 1:20;
[F] = ndgrid(Kset);
% Run a fitting on every pair fittingfunction(F(J,K), S(J,K))
fitresult = arrayfun(@(p1) myTuningWKNN(p1, Input, Target, TestInput, TestTarget, nDataset), F); 
[~, i1] = min(fitresult);
K = Kset(i1);
end

function WKNNError = myTuningWKNN(K, Input, Target, TestInput, TestTarget, nDataset)
WKNNPred = WKNNRegressor(TestInput, Input, Target, K);
switch nDataset
    case 1
        WKNNError =  mean(abs(WKNNPred(:)-TestTarget(:)));
    case 2
        [WKNNError, ~] = JointError(WKNNPred, TestTarget);
end
end