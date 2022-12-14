%% GP baseline model
% [m,s2]=gpbaseline(x,y,z)
% input:
% x        training instances
% y        training response
% z        test instances
% output
% m        predicted response on test instances
% s2       variance on each prediction

function [m,s2] = mygpbaseline(x,y,z)

meanfunc = {@meanSum, {@meanLinear, @meanConst}};
covfunc = {'covSum', {'covLINone','covConst','covNoise','covSEiso'}};
% meanfunc = {@meanConst};
likfunc = @likGauss;
% covfunc = {'covSEiso'};
hyp.cov = zeros(eval(feval(covfunc{:})),1); hyp.mean = zeros(size(x,2)+1,1); hyp.lik = log(0.1);
hyp = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, likfunc, x, y);
[m, s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

end