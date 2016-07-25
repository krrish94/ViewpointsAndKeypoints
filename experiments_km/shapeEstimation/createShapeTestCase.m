%% Creates a shape test case, given sequence num, frame num, and car id

% We're intereseted only in the 'car' class
class = 'car';

% Turn off Matlab warnings
warning('off', 'all');

% Declare global variables
globals;

% Add KITTI's Matlab code directory to path (the visualization one)
addpath /home/km/code/ViewpointsAndKeypoints/data/KITTI/devkit_tracking/matlab/


%% Parameters for KITTI (test data)

% ID of the sequence to be processed
% sequenceNum = 4;
sequenceNum = shapeTestCaseParams.seqNum;

% ID of the  image to process (in the sequence specified)
% imageId = 0;
imageId = shapeTestCaseParams.curFrameNum;
% ID(s) of the car to track
% carId = 0;
carId = shapeTestCaseParams.curCarId;

% Mode ('manual', or 'auto'). Specifies if the user will input the bounding
% box or if they have to be picked up from the ground truth.
bboxMode = 'auto';

% Base directory (containing KITTI data)
kittiBaseDir = fullfile(basedir, 'data', 'KITTI');
% Root directory containing KITTI images (for training sequences)
kittiImageDir = fullfile(kittiBaseDir, sprintf('image_02/%04d', sequenceNum));
% Directory containing KITTI labels (for training sequences)
kittiLabelDir = fullfile(kittiBaseDir, 'label_02');
% Directory containing camera calibration data
kittiCalibDir = fullfile(kittiBaseDir, 'calib');

% Get number of images in the sequence
numFrames = length(dir(fullfile(kittiImageDir)))-2;

% Get calibration parameters
% parameters: calib directory, sequence num, camera id (here, 2)
P = readCalibration(kittiCalibDir, sequenceNum, 2);

% Load labels for the current sequence
tracklets = readLabels(kittiLabelDir, sequenceNum);

% Create a cell to store data structs
dataStructs = {};

% Number of car detections thus far
numDetections = 0;

% Whether or not to save extracted features
saveKpFeats = false;


%% Create and save the test case

    
% Generate the file path for the current image to be processed
imgFile = fullfile(kittiImageDir, sprintf('%06d.png',imageId));
% Load the image
img = imread(imgFile);
    
% Tracklet for the current frame (usually comprises of multiple
% annotations, again referred to as tracklets)
tracklet = tracklets{imageId+1};

for j = 1:length(tracklet)
    % Current tracklet (annotation corresponding to a detection)
    curTracklet = tracklet(j);
    if ~strcmp(curTracklet.type, 'Car') && ~strcmp(curTracklet.type, 'Van')
        continue
    end
    
    % We only have to track specific cars. So, check if
    % the car id of the current tracklet is present in the list of
    % car ids to be tracked
    if curTracklet.id ~= carId;
        continue
    end
    
    %% Initialize the data struct
    
    % Get the bounding box (x1,y1,x2,y2), required by the CNN
    bbox = single([curTracklet.x1, curTracklet.y1, curTracklet.x2, curTracklet.y2]);
    % Get the bounding box (x,y,w,h), required by the rect command
    bboxPos = int16([curTracklet.x1, curTracklet.y1, (curTracklet.x2-curTracklet.x1+1), (curTracklet.y2-curTracklet.y1+1)]);
    % Determine whether or not the object is occluded
    occluded = curTracklet.occlusion;
    % Determine whether or not the object is truncated
    truncated = curTracklet.truncation;
    
    if truncated ~= 0 || occluded ~= 0
        continue
    end
    
    numDetections = numDetections + 1;
    
    % Create the data structure for the current detection
    curDataStruct.bbox = bbox;
    curDataStruct.fileName = imgFile;
    curDataStruct.labels = single(pascalClassIndex(class));
    curDataStruct.carId = curTracklet.id;
    % Image number in the sequence
    curDataStruct.imgNo = imageId;
    % Detection number in the image
    curDataStruct.detNo = j;
    
    
    %% Viewpoint features
    
    % Initialize the network for viewpoint prediction
    disp('Initialize viewpoint net');
    initViewpointNet;
    
    % Get pose prediction
    disp('Computing viewpoint features');
    poseFeat = runNetOnce(cnn_model, curDataStruct);
    poseFeat = poseFeat';
    predictedYaw = getPoseFromFeat(poseFeat);
    
    
    %% Coarse keypoint features
    
    disp('Computing coarse keypoint features');
    
    % Initialize the network used for coarse keypoint estimation
    initCoarseKeypointNet;
    
    % Run the network on the detection
    conv12Feat = runNetOnce(cnn_model_conv12Kps, curDataStruct);
    % Extract only the part of the feature vector corresponding to the
    % 'car' class
    conv12Feat = conv12Feat(7777:9792);
    conv12Feat = conv12Feat';
    
    
    %% Fine keypoint features
    
    disp('Computing fine keypoint features');

    % Initialize the network used for coarse keypoint estimation
    initKeypointNet;
    
    % Run the network on the detection
    conv6Feat = runNetOnce(cnn_model_conv6Kps, curDataStruct);
    % Extract only the part of the feature vector corresponding to the
    % 'car' class
    conv6Feat = conv6Feat(1945:2448);
    conv6Feat = conv6Feat';
    
    
    %% Load Pascal 3D data structs (used in computing viewpoint priors)
    
    % Set the name of the keypoint network
    params.kpsNet = 'vgg';
    
    % Load conv6 features (data struct, feat vector)
    
    if ~exist('featConv6', 'var')
        disp('Loading conv6');
        load(fullfile(cachedir, 'rcnnPredsKps', [params.kpsNet 'Conv6Kps'], class));
        % Flip the X and Y components of each heat map, and concatenate it back to
        % a row vector
        feat = flipMapXY(feat, [6 6]);
        % Resize the heatmap to the dimensions specified by params.heatMapDims
        feat6 = resizeHeatMap(feat, [6 6]);
        % Compute a softmax over the feature vector
        featConv6 = 1./(1+exp(-feat6));
    end
    
    
    % Load conv12 features
    
    if ~exist('featConv12', 'var')
        disp('Loading conv12');
        load(fullfile(cachedir, 'rcnnPredsKps', [params.kpsNet 'Conv12Kps'], class));
        % Flip the X and Y components of each heat map, and concatenate it back to
        % a row vector
        feat = flipMapXY(feat, [12 12]);
        % Resize the heatmap to the dimesnions specified by params.heatMapDims
        feat12 = resizeHeatMap(feat, [12 12]);
        % Compute a softmax over the feature vector
        featConv12 = 1./(1+exp(-feat12));
    end
    
    
    %% Compute pose prior heatmaps for each detection
    
    disp('Computing pose prior heatmaps');
    
    % Number of keypoints for the 'car' class
    numKps = params.numKps;
    
    % Loading rotation data for Pascal VOC, if it does not already exist
    if ~exist('rData', 'var')
        rData = load(fullfile(rotationPascalDataDir,class));
        rData = rData.rotationData;
    end
    % Extracting training samples and augmenting keypoints
    if ~exist('trainData', 'var')
        trainData = rData(ismember({rData(:).voc_image_id}, trainIds));
        % Note: the parameter passed to the following function isn't the
        % 'dataStructs' variable we create, but the 'dataStruct' variable that
        % contains ground truth annotations from Pascal/Imagenet
        trainData = augmentKps(trainData, dataStruct);
    end
    
    % Convert the pose prediction to a rotation matrix
    predTemp = predsToRotation(poseFeat);
    predTemp = predTemp{1};
    % Compute the pose prior heatmaps
    pFeat = neighborMapsKpsScore(predTemp, curDataStruct.bbox, trainData);
    posePriorFeat = pFeat;
    
    
    %% Concatenate all features
    
    disp('Concatenating features');
    
    % Upsample conv6 features to the specified size
    conv6Feat = flipMapXY(conv6Feat, [6 6]);
    conv6Feat = resizeHeatMap(conv6Feat, [6 6]);
    % % Perform exponential normalization to normalize the feature maps
    % conv6Feats = 1./(1+exp(-conv6Feats));
    
    % Repeat the process for conv12 features
    conv12Feat = flipMapXY(conv12Feat, [12 12]);
    conv12Feat = resizeHeatMap(conv12Feat, [12 12]);
    % conv12Feats = 1./(1+exp(-conv12Feats));
    
    % Compose a feature map using all features (conv + pose)
    featAll = 1./(1+exp(-conv6Feat-conv12Feat-log(posePriorFeat+eps)));
    
    
    %% Compute likelihood maps
    
    disp('Computing likelihood maps');
    likelihoodMaps = computeKpLikelihoodMaps(curDataStruct, featAll);
    
    
    %% Save the test case
    save(sprintf('cachedir/shapeTestCases/seq%02d_frame%03d_car%03d', sequenceNum, imageId, carId), 'featAll', 'likelihoodMaps', 'curDataStruct', 'curTracklet', 'predictedYaw')
    
    
end
