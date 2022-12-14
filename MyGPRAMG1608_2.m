% GPR with ABGMM features, proposed estimated ground truth

clear; clc; close all;

%% Load the data
load('AMG1608_MoodData.mat');
numTotal = 1608;
T = fieldnames(AMG1608_MoodData);
Feats = [1,2,3];
Labels = [5,8,10,11,13,14];
numFeats = length(Feats); numLabels = length(Labels);

GPR_R2_Test = zeros(numFeats,numLabels); GPR_MSE_Test = zeros(numFeats,numLabels); 
trn_samples = 1008; test_samples = 600;
% trn_samples = 300; test_samples = 100;
for j = 1:numLabels
    Y = AMG1608_MoodData.(T{Labels(j)});
    rp = randperm(numTotal);
    Y = Y(rp,:);
    for i = 1:numFeats
        X = AMG1608_MoodData.(T{Feats(i)});
        X = X(rp,:); 
        sT = sprintf('Feature %d: %s, Target %d: %s',i,T{Feats(i)},j,T{Labels(j)});
        disp(sT);
        % GPR-AVG Baseline
        trn_data.X = X(1:trn_samples, :);
        trn_data.y = Y(1:trn_samples, :);
        Xt = X(trn_samples+1:trn_samples+test_samples, :);
        Yt = Y(trn_samples+1:trn_samples+test_samples, :);
        [mean_y, ~] = mygpbaseline(trn_data.X,trn_data.y,Xt);
        [GPR_R2_Test(i,j), GPR_MSE_Test(i,j)] = rsquare(Yt, mean_y);
        fprintf('GPR_R2_Test = %f, GPR_MSE_Test = %f\n',GPR_R2_Test(i,j), GPR_MSE_Test(i,j));
    end
end

fprintf('Done!\n');
% Save the results
save('AMG1608_GPR_Results_2.mat','GPR_R2_Test', 'GPR_MSE_Test');
