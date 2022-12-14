clear; clc; close all;

%% Load the data
load('MyMoodDataAMG1608_X_Y.mat');
numTotal = 1608;

% 12 feature types
% 10 target variables (14, 15, 17, 18, 20, 21, 23, 24, 26, 27)

T = fieldnames(AMG1608MoodData);
numFeats = 12; % 1:12 fields of structure
numLabels = 10;
% Labels = [14, 15, 17, 18, 20, 21, 23, 24, 26, 27];
% Feats = [1,2,3,4,5,6,7,8,9,10,11,12];
Feats = [2,3,8,9];
Labels = [14, 17];
numFeats = 4; numLabels = 2;
% Labels = [17];%, 18, 20, 21, 23, 24, 26, 27];
% 14 = Val_Avg, 15 = Val_Med, 17 = Aro_Avg, 18 = Aro_Med, 20 = Theta_Avg,
% 21 = Theta_Med, 23 = Rho_Avg, 24 = Rho_Med, 26 = TD_Avg, 27 = TD_Med

GPR_R2_Test = zeros(numFeats,numLabels); GPR_MSE_Test = zeros(numFeats,numLabels); 
% trn_samples = 1008; test_samples = 600;
trn_samples = 300; test_samples = 100;
for j = 1:numLabels
    Y = AMG1608MoodData.(T{Labels(j)});
    rp = randperm(numTotal);
    Y = Y(rp,:);
    for i = 1:numFeats
%     sF = sprintf('Feature %d: %s',i,T{i});
%     disp(sF);
        X = AMG1608MoodData.(T{Feats(i)});
        X = X(rp,:); 
        sT = sprintf('Feature %d: %s, Target %d: %s',i,T{Feats(i)},j,T{Labels(j)});
        disp(sT);
        % GPR-AVG Baseline
        trn_data.X = X(1:trn_samples, :);
        trn_data.y = Y(1:trn_samples, :);
        Xt = X(trn_samples+1:trn_samples+test_samples, :);
        Yt = Y(trn_samples+1:trn_samples+test_samples, :);
%         if i==1
%             trn_data.X = zscore(trn_data.X); Xt = zscore(Xt);
%         end
        [mean_y, ~] = mygpbaseline(trn_data.X,trn_data.y,Xt);
        [GPR_R2_Test(i,j), GPR_MSE_Test(i,j)] = rsquare(Yt, mean_y);
        fprintf('GPR_R2_Test = %f, GPR_MSE_Test = %f\n',GPR_R2_Test(i,j), GPR_MSE_Test(i,j));
    end
end

fprintf('Done!\n');
% Save the results
save('AMG1608_GPR_Results.mat','GPR_R2_Test', 'GPR_MSE_Test');
