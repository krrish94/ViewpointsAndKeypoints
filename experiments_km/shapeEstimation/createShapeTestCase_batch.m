%% Creates shape test cases, given sequences, frames, and car ids
% For now, the car id functionality hasn't been implemented. We create test
% cases for all non-occluded, non-truncated cars in the frames specified.


%% User configurable parameters

% List of sequences to be used
sequences = {2, 1, 3, 4};
% Frames of the sequence for which test cases have to be generated
% frames = {[0], [0], [0], [0], [0], [0]};
frames = {[74, 87, 98], [57, 66, 72, 74, 94, 112, 234, 250, 261, 370, 385], [9, 20, 41, 133], [197, 246]};


%% Declaring a few variables

% We're intereseted only in the 'car' class
class = 'car';

% Turn off Matlab warnings
warning('off', 'all');

% Declare global variables
globals;

% Add KITTI's Matlab code directory to path (the visualization one)
addpath /home/km/code/ViewpointsAndKeypoints/data/KITTI/devkit_tracking/matlab/

% Base directory (containing KITTI data)
kittiBaseDir = fullfile(basedir, 'data', 'KITTI');
% Directory containing KITTI labels (for training sequences)
kittiLabelDir = fullfile(kittiBaseDir, 'label_02');
% Directory containing camera calibration data
kittiCalibDir = fullfile(kittiBaseDir, 'calib');


%% Main processing loop

% For each sequence to be processed
for i = 1:length(sequences)
    
    curSeqNum = sequences{i};
    
    % Root directory containing KITTI images (for training sequences)
    kittiImageDir = fullfile(kittiBaseDir, sprintf('image_02/%04d', curSeqNum));
    
    % Get calibration parameters
    % parameters: calib directory, sequence num, camera id (here, 2)
    P = readCalibration(kittiCalibDir, curSeqNum, 2);
    
    % Load labels for the current sequence
    tracklets = readLabels(kittiLabelDir, curSeqNum);
    
    % Create a cell to store data structs
    dataStructs = {};
    
    % Number of detections so far (in the present sequence)
    numDetections = 0;
    
    % For each frame in the sequence to be processed
    frameList = frames{i};
    for j = 1:length(frameList)
        
        curFrameNum = frameList(j);
        fprintf('Processing seq:%02d frame:%03d ...\n', curSeqNum, curFrameNum);
        
        % Load the tracklet corresponding to the current frame
        tracklet = tracklets{j+1};
        
        % For each detection in the tracklet
        for k = 1:length(tracklet)
            
            curTracklet = tracklet(k);
            
            if ~strcmp(curTracklet.type, 'Car') && ~strcmp(curTracklet.type, 'Van')
                continue
            end
            
            % Get the bounding box (x1,y1,x2,y2), required by the CNN
            bbox = single([curTracklet.x1, curTracklet.y1, curTracklet.x2, curTracklet.y2]);
            % Get the bounding box (x,y,w,h), required by the rect command
            bboxPos = int16([curTracklet.x1, curTracklet.y1, (curTracklet.x2-curTracklet.x1+1), (curTracklet.y2-curTracklet.y1+1)]);
            % Determine whether or not the object is occluded
            occluded = curTracklet.occlusion;
            % Determine whether or not the object is truncated
            truncated = curTracklet.truncation;
            % Generate the file path for the current image to be processed
            imgFile = fullfile(kittiImageDir, sprintf('%06d.png',curFrameNum));
            
            if truncated ~= 0 || occluded ~= 0
                continue
            end
            
            numDetections = numDetections + 1;
            
            
            %% Initialize the data struct
            
            % Create the data structure for the current detection
            curDataStruct.bbox = round(bbox);
            curDataStruct.fileName = imgFile;
            curDataStruct.labels = single(pascalClassIndex(class));
            curDataStruct.carId = curTracklet.id;
            % Sequence number
            curDataStruct.seqNum = curSeqNum;
            % Image number in the sequence
            curDataStruct.imgNo = curFrameNum;
            % Detection number in the image
            curDataStruct.detNo = k;
            curDataStruct.curTracklet = curTracklet;
            
            % Append this struct to the dataStructs cell array
            dataStructs{numDetections} = curDataStruct;
            
        end
        
    end
    
    
    %% Viewpoint features
    
    % Initialize the network for viewpoint prediction
    disp('Initializing viewpoint net ...');
    initViewpointNet;
    
    % Initialize an array to store pose features
    poseFeats = zeros(length(dataStructs), 84);
    
    % For each detection
    disp('Computing viewpoint features ...');
    for j = 1:length(dataStructs)
        curDataStruct = dataStructs{j};
        curPoseFeat = runNetOnce(cnn_model, curDataStruct);
        poseFeats(j,:) = curPoseFeat';
        dataStructs{j}.predictedYaw = getPoseFromFeat(curPoseFeat);
    end
    
    
    %% Coarse keypoint features
    
    disp('Initializing coarse keypoint net ...');
    
     % Initialize the network used for coarse keypoint estimation
    initCoarseKeypointNet;
    
    % Initialize an array to store coarse keypoint features (each of them
    % is a 12 x 12 map; there are 14 such maps, so 2016 dimensions in all)
    conv12Feats = zeros(length(dataStructs), 12*12*14);
    
    % For each detection
    disp('Computing coarse keypoint features ...');
    for j = 1:length(dataStructs)
        % Run the network on the detection
        curConv12Feat = runNetOnce(cnn_model_conv12Kps, curDataStruct);
        % Extract only the part of the feature vector corresponding to the
        % 'car' class
        curConv12Feat = curConv12Feat(7777:9792);
        conv12Feats(j,:) = curConv12Feat';
    end
    
    
    %% Fine keypoint features
    
    disp('Initializing fine keypoint net ...');
    
     % Initialize the network used for coarse keypoint estimation
    initKeypointNet;
    
    % Initialize an array to store coarse keypoint features (each of them
    % is a 6 x 6 map; there are 14 such maps, so 504 dimensions in all)
    conv6Feats = zeros(length(dataStructs), 6*6*14);
    
    % For each detection
    disp('Computing fine keypoint features ...');
    for j = 1:length(dataStructs)
        % Run the network on the detection
        curConv6Feat = runNetOnce(cnn_model_conv6Kps, curDataStruct);
        % Extract only the part of the feature vector corresponding to the
        % 'car' class
        curConv6Feat = curConv6Feat(1945:2448);
        conv6Feats(j,:) = curConv6Feat';
    end
    
    
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
    
    % Matrix to store pose priors
    posePriorFeats = zeros(length(dataStructs), params.heatMapDims(1)*params.heatMapDims(2)*params.numKps);
    
    % For each detection
    for j = 1:length(dataStructs)
        % Convert the pose prediction to a rotation matrix
        predTemp = predsToRotation(poseFeats);
        predTemp = predTemp{1};
        % Compute the pose prior heatmaps
        pFeat = neighborMapsKpsScore(predTemp, dataStructs{j}.bbox, trainData);
        posePriorFeats(j,:) = pFeat;
    end
    
    
    %% Concatenate all features
    
    disp('Concatenating features');

    % Upsample conv6 features to the specified size
    conv6Feats = flipMapXY(conv6Feats, [6 6]);
    conv6Feats = resizeHeatMap(conv6Feats, [6 6]);
    % % Perform exponential normalization to normalize the feature maps
    % conv6Feats = 1./(1+exp(-conv6Feats));
    
    % Repeat the process for conv12 features
    conv12Feats = flipMapXY(conv12Feats, [12 12]);
    conv12Feats = resizeHeatMap(conv12Feats, [12 12]);
    % conv12Feats = 1./(1+exp(-conv12Feats));
    
    % Compose a feature map using all features (conv + pose)
    featAll = 1./(1+exp(-conv6Feats-conv12Feats-log(posePriorFeats+eps)));
    
    
    %% Compute likelihood maps
    
    % For each detection
    disp('Computing likelihood maps');
    for j = 1:length(dataStructs)
        likelihoodMap = computeKpLikelihoodMaps(dataStructs{j}, featAll(j,:));
        dataStructs{j}.likelihoodMap = likelihoodMap;
        dataStructs{j}.featAll = featAll(j,:);
    end
    
    
    %% Save test cases
    
    for j = 1:length(dataStructs)
        curDataStruct = dataStructs{j};
        save(sprintf('cachedir/shapeTestCases/seq%02d_frame%03d_car%03d', dataStructs{j}.seqNum, dataStructs{j}.imgNo, dataStructs{j}.carId), 'curDataStruct')
    end
    
    
end











%% Old code (very inefficient, but works)

% % First frame to be processed for each sequence specified
% startFrames = {0};
% % Last frame to be processed for each sequence specified
% endFrames = {446};
% % List of cars to be processed for each sequence specified
% carList = {[1:9, 11:13, 17:19, 23, 28, 33:34, 36:41, 43, 46, 49, 52, 54, 56:57, 59, 63, 67, 69, 71, 75, 81, 83:84, 86, 88, 90:92, 96:97]};


% Get sequence information corresponding to all cars in all sequences,
% where the cars are visible for more than 50 frames
% -- TODO: Implement this functionality -- %


% % Sanity checks
% if length(sequences) ~= length(startFrames)
%     disp('Need equal number of start frames as sequences');
%     return;
% elseif length(sequences) ~= length(endFrames)
%     disp('Need equal number of end frames as sequences');
%     return;
% elseif length(startFrames) ~= length(endFrames)
%     disp('Need equal number of end frames and start frames')
%     return;
% end
% 
% if length(carList) ~= length(sequences)
%     disp('Need equal number of car lists and sequences');
% end

% % For each sequence in the list
% for i = 1:length(sequences)
%     shapeTestCaseParams.seqNum = sequences{i};
%     disp(sprintf('Processing seq: %d', sequences{i}));
%     if endFrames{i} - startFrames{i} < 0
%         disp('End frame cannot occur after start');
%         disp(sprintf('Error in seq %d, start: %d, end: %d', sequences{i}, startFrames{i}, endFrames{i}));
%         return;
%     end
%     
%     % Variables for displaying progress
%     reverseStr = '';
%     numFrames = 0;
%         
%     curCarList = carList{i};
%     % For each frame in the sequence
%     for j = startFrames{i}:endFrames{i}
%         numFrames = numFrames + 1;
%         if mod(numFrames,10) == 0 && startFrames{i} ~= endFrames{i}
%             percentDone = numFrames*100/(startFrames{i}-endFrames{i});
%             msg = sprintf('Done: %3.1f', percentDone);
%             fprintf([reverseStr, msg]);
%             reverseStr = repmat(sprintf('\b'), 1, length(msg));
%         end
%         
%         shapeTestCaseParams.curFrameNum = j;
%         % For each car to be processed
%         for k = 1:length(curCarList)
%             shapeTestCaseParams.curCarId = curCarList(k);
%             % Run the script that creates the test case
%             createShapeTestCase;
%         end
%     end
% end
