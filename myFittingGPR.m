function GPError = myFittingGPR(kparam, lambda, Input, Target, TestInput, TestTarget)

K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
GPError =  mean(abs(GPPred(:)-TestTarget(:)));
