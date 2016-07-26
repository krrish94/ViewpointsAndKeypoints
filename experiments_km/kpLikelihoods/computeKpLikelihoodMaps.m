function [likelihoodMaps] = computeKpLikelihoodMaps(curDataStruct, curFeat)
% COMPUTEKPLIKELIHOODS  Function to compute the keypoint likelihoods, given
% a data struct and the corresponding feature vector.
%
%   Inputs:
%   -------
%   curDataStruct: data struct for the current detection
%   curFeat: feature vector for the current detection


% Declaring global variables
globals;

% Get relevant data from the data struct
bbox = curDataStruct.bbox;
fileName = curDataStruct.fileName;

% % Get keypoint coordinates and scores
% [kpCoords, scores] = maxLocationPredict(curFeat, bbox, params.heatMapDims);

% Read in image
im = imread(fileName);
% Round the bbox to integral vertices
bbox = round(bbox);

% Get individual heatmaps from the feature vector
heatMaps = reshape(curFeat, [params.heatMapDims(1), params.heatMapDims(2), params.numKps]);

% Initialize the likelihoods matrix
likelihoodMaps = zeros([size(im), params.numKps]);

% For each keypoint
for kpIdx = 1:params.numKps
    
    % To avoid indexing errors, wherever bbox is 0, make it 1. (Making this
    % change after encountering such an error)
    bbox(bbox == 0) = 1;

    % Scale the heatmap for the first keypoint to fit the bounding box. Size of
    % an image is specified in terms of height x width
    resizedHeatMap = imresize(heatMaps(:,:,kpIdx), [bbox(4) - bbox(2) + 1, bbox(3) - bbox(1) + 1]);
    
    % Replace the bounding box with the heatmaps
    likelihoodMaps(bbox(2):bbox(4), bbox(1):bbox(3), :, kpIdx) = repmat(resizedHeatMap, 1, 1, 3);

end


end
