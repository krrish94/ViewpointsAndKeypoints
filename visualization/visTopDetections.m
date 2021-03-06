function [] = visTopDetections(class)
%VISTOPDETECTIONS Summary of this function goes here
%   Detailed explanation goes here


% Declaring global variables
globals;
% Name of the directory where detection results are stored
proto = 'vggJointVps';suff = '';
dataStructsDir = fullfile(cachedir,['rcnnDetectionPredsVps'],[proto suff]);
% Load the detections
load(fullfile(dataStructsDir,'allDets.mat'));
% Load detection data structs corresponding to the class of interest
cInd = pascalClassIndex(class);
cands = dataStructs(cInd);

% Get the names of the validation images
pascalValNamesFile = fullfile(cachedir,'voc_val_names.mat');
valNames = load(pascalValNamesFile);
valNames = valNames.val_names;
cands.voc_ids = cell(size(cands.boxes));

% Path to directory containing CAD models
CADPath = fullfile(PASCAL3Ddir,'CAD',class);
% Load the CAD models available for the current class
cad = load(CADPath);
cad = cad.(class);
% Get the vertices and faces from the first CAD model
vertices = cad(1).vertices;
faces = cad(1).faces;

% For each validation instance, create a voc_ids field in the cands struct
for i = 1:length(valNames)
    cands.voc_ids{i} = valNames(i*ones(size(cands.boxes{i},1),1));
end

% Concatenate all bboxes
boxes = vertcat(cands.boxes{:});
% The fifth entry of each bbox detection corresponds to the score. Get
% scores for all detections.
scores = boxes(:,5);
% Sort scores in the descending order
[scores,perm] = sort(scores,'descend');

% Store bboxes in the descending order of their corresponding scores
boxes = boxes(perm,1:4);
% Do the same for extracted features
feat = vertcat(cands.feat{:});
feat = feat(perm,:);
% Do the same for their voc_ids
ids = vertcat(cands.voc_ids{:});
ids = ids(perm);

% Get pose predictions
preds = poseHypotheses(feat,1,0);
preds = preds{1};
% Convert them to Euler angle encoding
eulersPred = decodePose(preds,params.angleEncoding);

rotX = diag([1 -1 -1]);

% Visualize the top 100 detections
%for i=1:length(ids)
for i = 1:100
    % Display the voc_id of the current detection
    disp( ids{i})
    % Read in the image
    im = imread([pascalImagesDir '/' ids{i} '.jpg']);
    % In one subplot, plot the image and the bbox
    subplot(1,2,1);
    showboxes(im,boxes(i,:));
    axis image;
    
    % Create another subplot
    subplot(1,2,2);
    
    % Get the Euler angles for the current prediction
    euler = eulersPred(i,:);
    % Compute the transformation from the CAD frame to the viewpoint
    R = angle2dcm(euler(1), euler(2)-pi/2, -euler(3),'ZXZ');
    R = rotX*R';
    % Apply the transformation to the vertices of the CAD model
    verticesP = R*vertices';
    verticesP = verticesP';
    % Plot the transformed CAD model
    trisurf(faces,verticesP(:,1),verticesP(:,2),verticesP(:,3));axis equal;view(0,-90);
    % Use the score as the title of the plot
    title(num2str(scores(i)));
    % Wait for key press from the user
    pause();
    % Close the current figure. Opens a new figure for the next detection.
    close all;
end

end

