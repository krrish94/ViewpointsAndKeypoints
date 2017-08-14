%% Script to predict keypoints for the chair class

class = 'chair';

% Turn off Matlab warnings
warning('off', 'all');

% Declare global variables
globals;

% Initialize the network for viewpoint prediction
initViewpointNet;

% Read in the image
imgFile = '/home/km/code/Object_SLAM_ceres/data/video4/frame_00038.jpg';
img = imread(imgFile);

imshow(img);
r = imrect;
position = wait(r);
bbox = single([position(1), position(2), position(1)+position(3), position(2)+position(4)]);

% Create the data structure for the current detection
dataStruct.bbox = bbox;
dataStruct.fileName = imgFile;
dataStruct.labels = single(pascalClassIndex(class));

% Run the viewpoint network on the detection
initViewpointNet;
featVec = runNetOnce(cnn_model, dataStruct);
yaw = getPoseFromFeat_test(featVec);

disp('Loading conv6');
initKeypointNet;
featVec_6Kps = runNetOnce(cnn_model_conv6Kps, dataStruct);
featVec_temp = featVec_6Kps(2449:2808);
feat = flipFeatVecXY(featVec_temp, [6,6]);
feat6 = resizeHeatMapSingle(feat, [6 6], params.heatMapDims);
featConv6 = 1./(1+exp(-feat6));

disp('Loading conv12');
initCoarseKeypointNet;
featVec_12Kps = runNetOnce(cnn_model_conv12Kps, dataStruct);
featVec_temp = featVec_12Kps(9793:11232);
feat = flipFeatVecXY(featVec_temp, [12 12]);
feat12 = resizeHeatMapSingle(featVec_temp, [12 12], params.heatMapDims);
featConv12 = 1./(1+exp(-feat12));

posePriorFeat = computePosePriorsChair(dataStruct, featVec);

testFeat = 1./(1+exp(-feat6-feat12-posePriorFeat));
disp('Predicting keypoints');
[kpCoords,scores] = maxLocationPredict(featConv6, bbox, params.heatMapDims);
imshow(img);
hold on;
scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled');
