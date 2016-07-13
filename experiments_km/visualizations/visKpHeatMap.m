function [newIm] = visKpHeatMap(curDataStruct, curFeat)
% VISKPHEATMAP  Function to plot the keypoint heat map, given a data struct
% and the corresponding feature vector


% Declaring global variables
globals;

% Get relevant data from the data struct
bbox = curDataStruct.bbox;
fileName = curDataStruct.fileName;

% Get keypoint coordinates and scores
[kpCoords, scores] = maxLocationPredict(curFeat, bbox, params.heatMapDims);

% Read in image
im = imread(fileName);
% Round the bbox to integral vertices
bbox = round(bbox);

% Get individual heatmaps from the feature vector
heatMaps = reshape(curFeat, [params.heatMapDims(1), params.heatMapDims(2), params.numKps]);

% Scale the heatmap for the first keypoint to fit the bounding box. Size of
% an image is specified in terms of height x width
resizedHeatMap = imresize(heatMaps(:,:,1), [bbox(4) - bbox(2) + 1, bbox(3) - bbox(1) + 1]);

% Initialize a new image of same size as the original image
newIm = zeros(size(im));
% Replace the bounding box with the heatmaps
newIm(bbox(2):bbox(4), bbox(1):bbox(3), :) = repmat(resizedHeatMap, 1, 1, 3);

% Display the heatmap for the first keypoint
imshow(newIm);
hold on;
colormap('hot');
im = imagesc(newIm(:,:,1));

end
