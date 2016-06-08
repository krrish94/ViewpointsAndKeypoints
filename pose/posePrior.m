function [priorFeat] = posePrior(dataStruct,class,trainIds)
%POSEPRIOR Summary of this function goes here
%   Detailed explanation goes here

% Declaring global variables
globals;
% Loading rotation data for Pascal VOC
rData = load(fullfile(rotationPascalDataDir,class));
rData = rData.rotationData;
% Extracting training samples
trainData = rData(ismember({rData(:).voc_image_id},trainIds));
trainData = augmentKps(trainData,dataStruct);

% Loading viewpoint predictions for training samples
% preds = load(fullfile(cachedir,'rcnnPredsVpsKeypoint',params.features,class));
preds = load(fullfile(cachedir,'rcnnPredsVps',params.features,class));
% Get the feature vector and transform it to a rotation matrix
preds = preds.feat;
preds = predsToRotation(preds);

% Get the height and width of the heatmap
H = params.heatMapDims(2);
W = params.heatMapDims(1);
% Number of keypoints for the current class
Kp = size(trainData(1).kps,1);
% Number of training images
N = length(dataStruct.voc_image_id);
% Initialize a matrix to store prior features
priorFeat = zeros(N,H*W*Kp);
% Bounding boxes
bboxes = dataStruct.bbox;

% For each image
for i = 1:N
    % Compute the prior score
    pFeat = neighborMapsKpsScore(preds{i},bboxes(i,:),trainData);    
    priorFeat(i,:) = pFeat;
end

end