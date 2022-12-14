% loads HoG featurs, predict 3d poses on test set, and exports the pose vector to XML file.
% @he_dataset and @body_pose contained in Source Code v. 1.1 (beta) is needed to run this function, 
% available at http://vision.cs.brown.edu/humaneva/download1.html

clear all
Param.kparam1 = 1e-5/3;
Param.kparam2 = 5*1e-6;
Param.lambda = 1e-3;
Param.knn = 100;

% parameters
savedir = './XML/';
% load HMAX features
Train = load('./data/HMAX_S1_Walking_TrainValidation_C1C2C3.mat');
Input = Train.hmax;
Target = Train.pose;
Test = load('./data/HMAX_S1_Walking_Test_C1C2C3.mat');
TestInput = Test.hmax;

% make the prediction
Y = TGPKNN(TestInput, Input, Target, Param);
frames = size(Y,1);
dset = he_dataset('HumanEvaI', 'Test');

% print XML format
tic;
for FRAME = 1:frames

    pose_formatted = reshape(Y(FRAME, :), 3, 20);
    
    % make body_pose object
    body_poses(FRAME) = body_pose(  'torsoDistal',              pose_formatted(:, 1), ...
                                    'upperLLegProximal',        pose_formatted(:, 2), ...
                                    'upperLLegDistal',          pose_formatted(:, 3), ...
                                    'lowerLLegProximal',        pose_formatted(:, 4), ...
                                    'lowerLLegDistal',          pose_formatted(:, 5), ... 
                                    'upperRLegProximal',        pose_formatted(:, 6), ...
                                    'upperRLegDistal',          pose_formatted(:, 7), ...
                                    'lowerRLegProximal',        pose_formatted(:, 8), ...
                                    'lowerRLegDistal',          pose_formatted(:, 9), ...
                                    'torsoProximal',            pose_formatted(:, 10), ...
                                    'headProximal',             pose_formatted(:, 11), ...
                                    'headDistal',               pose_formatted(:, 12), ...
                                    'upperLArmProximal',        pose_formatted(:, 13), ...
                                    'upperLArmDistal',          pose_formatted(:, 14), ...
                                    'lowerLArmProximal',        pose_formatted(:, 15), ...
                                    'lowerLArmDistal',          pose_formatted(:, 16), ...
                                    'upperRArmProximal',        pose_formatted(:, 17), ...
                                    'upperRArmDistal',          pose_formatted(:, 18), ...
                                    'lowerRArmProximal',        pose_formatted(:, 19), ...
                                    'lowerRArmDistal',          pose_formatted(:, 20));
    if (mod(FRAME, 10) == 0)
        fprintf('frame: %d    time: %d\n', FRAME, toc);
        tic
    end
end

% export to XML
filename = 'TGPKNN_HMAX_S1_Walking_Test_C1C2C3'; 
result = exportXML(body_poses, (1:frames)+6, '', dset(1), strcat(savedir, filename));
clear body_poses;

