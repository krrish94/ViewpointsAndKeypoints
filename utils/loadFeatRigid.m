%% Load conv6 features (data struct, feat vector)

disp('Loading conv6');
load(fullfile(cachedir, 'rcnnPredsKps', [params.kpsNet 'Conv6Kps'], class));
% Flip the X and Y components of each heat map, and concatenate it back to
% a row vector
feat = flipMapXY(feat, [6 6]);
% Resize the heatmap to the dimensions specified by params.heatMapDims
feat6 = resizeHeatMap(feat, [6 6]);
% Compute a softmax over the feature vector
featConv6 = 1./(1+exp(-feat6));


%% Load conv12 features

disp('Loading conv12');
load(fullfile(cachedir, 'rcnnPredsKps', [params.kpsNet 'Conv12Kps'], class));
% Flip the X and Y components of each heat map, and concatenate it back to
% a row vector
feat = flipMapXY(feat, [12 12]);
% Resize the heatmap to the dimesnions specified by params.heatMapDims
feat12 = resizeHeatMap(feat, [12 12]);
% Compute a softmax over the feature vector
featConv12 = 1./(1+exp(-feat12));


%% Load pose priors

% [priorFeat] = posePrior(dataStruct,class,fnamesTrain);
priorFeat = posePrior(dataStruct, class, trainIds);


%% Create composite feature maps

% Compose a feature map using all features (conv + pose)
featAll = 1./(1+exp(-feat6 - feat12 - log(priorFeat + eps)));

% Compose a feature map using pose features
featConv = 1./(1+exp(-feat6 - feat12));
