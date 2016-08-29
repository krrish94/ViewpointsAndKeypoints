% We're interested only in the 'car' class
class = 'car';

% Load pose prediction features, if they don't already exist
if ~exist('feat', 'var')
    load(fullfile(cachedir, 'rcnnPredsVps', 'vggJointVps', class), 'feat');
end
% The variable 'feat' now contains pose features.

% Load train/test data, if not already loaded
if ~exist('data', 'var')
    data = load(fullfile(cachedir,'evalSets',class));
end

% Generate evaluation sets from the data loaded above
[trainLabels,testLabels,trainFeats,testFeats] = generateEvalSetData(data);

% Predict poses on test set (alpha is the weight parameter, used to weight
% coarse and fine pose features; nHypotheses is the number of hypotheses to
% consider while predicting pose)
alphaOpt = params.alpha;
nHypotheses = params.nHypotheses;
testPreds = poseHypotheses(testFeats, nHypotheses, alphaOpt);
% Since we consider only one hypothesis, load it
testPreds = testPreds{1};


% Generate a sample of 1 image to test the visualizeRotations function

% Index (indices) of the current image(s) we're considering
% curIdx = 1:10;
curIdx = 11:20;

% Ground truth for the current instance(s)
gt = testLabels(curIdx,:);
% Prediction for the current instance(s)
prediction = testPreds(curIdx,:);
% Data struct for the current instance(s)
for i = 1:length(curIdx)
    dataStruct_temp.voc_ids{i} = data.test.voc_ids{curIdx(i)};
    dataStruct_temp.bboxes(i,:) = data.test.bboxes(curIdx(i),:);
end

% Run the 'visualizeRotations' function. Not so useful, as this just plots
% the test image, with the predicted and ground truth azimuths, as well as
% their ratio.
% visualizeRotations(gt, prediction, dataStruct_temp, 'euler', 1);

%% Visualize CAD models of the predictions

% Path to directory containing CAD models
CADPath = fullfile(PASCAL3Ddir,'CAD',class);
% Load the CAD models available for the current class
cad = load(CADPath);
cad = cad.(class);
% Get the vertices and faces from the first CAD model
vertices = cad(1).vertices;
faces = cad(1).faces;

% Rotation matrix commonly applied to all instances (why ???)
rotX = diag([1 -1 -1]);

% For each prediction (instance)
for i = 1:length(curIdx)
    % Display the voc_id of the current instance
    disp(dataStruct_temp.voc_ids{i});
    % Read in the image
    im = imread(fullfile(pascalImagesDir, [dataStruct_temp.voc_ids{i}, '.jpg']));
    
    % In one subplot, plot the image and the corresponding bbox
    subplot(1,2,1);
    showboxes(im, dataStruct_temp.bboxes(i,:));
    axis image;
    
    % Get the Euler angles for the current prediction
    euler = testPreds(i,:);
    
    % Compute the transformation from the CAD frame to the viewpoint
    % R = angle2dcm(euler(1), euler(2)-pi/2, -euler(3),'ZXZ');
    % R = [1,0,0;0,0,-1;0,1,0]; % pi/2 about X
    % R = diag([1, -1, -1]); % pi about X
    % R = [cos(pi/4), -sin(pi/4), 0; sin(pi/4), cos(pi/4), 0; 0, 0, 1]*diag([1, -1, -1]); % pi about X, and pi/2 about Z
    % Apply the transformation to the vertices of the CAD model
    % R = rotX*R';
    R = [cos(-euler(3)), -sin(-euler(3)), 0; sin(-euler(3)), cos(-euler(3)), 0; 0, 0, 1]; % -euler(3) about Z
    R = [1, 0, 0; 0, cos(euler(2)-(pi/2)), -sin(euler(2)-(pi/2)); 0, sin(euler(2)-(pi/2)), cos(euler(2)-(pi/2))]*R; % euler(2)-pi/2 about X
    R = [cos(euler(1)), -sin(euler(1)), 0; sin(euler(1)), cos(euler(1)), 0; 0, 0, 1]*R; % euler(1) about Z
    R = rotX*R;
    verticesP = R*vertices';
    verticesP = verticesP';
    
    % In the other subplot, plot the CAD model, transformed according to
    % the estimated viewpoint of the current instance
    subplot(1,2,2);
    trisurf(faces,verticesP(:,1),verticesP(:,2),verticesP(:,3));axis equal;view(0,-90);
    
    % Wait for key press from the user
    pause();
    % Close the current figure. Opens a new figure for the next detection.
    close all;
end
