kparam = 1e-5;
lambda = 1e-4;
K = EvalKernel(Input,Input,'rbf',kparam);
alpha = (K+lambda*eye(size(K)))\Target;
testK = EvalKernel(TestInput,Input,'rbf',kparam);
GPPred = testK*alpha;
[GPError, GPErrorvec] = JointError(GPPred, TestTarget);
disp(['GP: ' num2str(GPError)]);