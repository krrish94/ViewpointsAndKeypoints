% Testing keypoint prediction on KITTI images


disp('Computing conv6 features');
initKeypointNet;
featVec_6Kps = runNetOnce(cnn_model_conv6Kps, dataStruct);
featVec_temp = featVec_6Kps(1945:2448);
feat = flipFeatVecXY(featVec_temp, [6,6]);
feat6 = resizeHeatMapSingle(feat, [6 6], params.heatMapDims);
featConv6 = 1./(1+exp(-feat6));

disp('Computing conv12 features');
initCoarseKeypointNet;
featVec_12Kps = runNetOnce(cnn_model_conv12Kps, dataStruct);
featVec_temp = featVec_12Kps(7777:9792);
feat = flipFeatVecXY(featVec_temp, [12 12]);
feat12 = resizeHeatMapSingle(featVec_temp, [12 12], params.heatMapDims);
featConv12 = 1./(1+exp(-feat12));

disp('Computing pose prior features');
posePriorFeat = computePosePriors(dataStruct, featVec);
featPose = resizeHeatMapSingle(posePriorFeat, [12, 12], params.heatMapDims);

testFeat = featConv6 + featConv12 + posePriorFeat;


disp('Predicting keypoints');

% % Read in image and get the bounding box
% 
% % imName = '000000.png';
% % bbox = [295, 170, 462, 290];
% 
% % imName = '000001.png';
% % bbox = [295, 170, 462, 290];
% 
% % im = imread(fullfile(basedir, 'data', 'KITTI', 'Seq00', imName));


im = imread(dataStruct.fileName);

% Predict keypoints and their scores
[kpCoords,scores] = maxLocationPredict(testFeat, bbox, params.heatMapDims);

% Get the first 14 keypoints
kpCoords = kpCoords(1:2,1:14);

bbox2(1) = bbox(1); bbox2(2) = bbox(2); bbox2(3) = bbox(3)-bbox(1); bbox2(4) = bbox(4)-bbox(2);
imshow(im);
hold on
scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled')
% scatter(kps(:,1),kps(:,2),50,'b','filled')
rectangle('Position', bbox2, 'LineWidth', 3, 'EdgeColor', 'g');
hold off














