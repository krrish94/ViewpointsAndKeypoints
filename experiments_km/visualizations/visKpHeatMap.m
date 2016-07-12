function [] = visKpHeatMap(curDataStruct, curFeat)
% VISKPHEATMAP  Function to plot the keypoint heat map, given a data struct
% and the corresponding feature vector


% Declaring global variables
globals;

% Get relevant data from the data struct
bbox = curDataStruct.bbox;
fileName = curDataStruct.fileName;

% Get keypoint coordinates and scores
[kpCoords, scores] = maxLocationPredict(curFeat, bbox, params.heatMapDims);

% Get keypoint coordinates relative to the top left corner of the bbox
x = kps(1,:) - bbox(1);
y = kps(2,:) - bbox(2);

% Read in image
im = imread(fileName);
% Round the bbox to integral vertices
bbox = round(bbox);



end
