%% Script to evaluate Viewpoint consistency over a KITTI sequence

% The aim of this experiment is to check if the predicted viewpoint
% (coarse, i.e., the 21-bin keypoint prediction) is consistent-enough that
% we can start using it for SLAM (after applying temporal priors or median
% filters)


% We're intereseted only in the 'car' class
class = 'car';

% Turn off Matlab warnings
warning('off', 'all');

% Declare global variables
globals;

% Add KITTI's Matlab code directory to path (in the tracking development
% kit provided by KITTI)
addpath /home/km/code/ViewpointsAndKeypoints/data/KITTI/devkit_tracking/matlab/


%% Parameters for KITTI (test data)

% ID of the sequence to be processed
% sequenceNum = 4;  % Uncomment if running this script independently

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

% ID of the first image to process (in the sequence specified)
startImageId = 0;
% ID of the last image to process (in the sequence specified)
% endImageId = 100;
endImageId = numFrames-1;
% Creating a list of images to process
imageList = startImageId:endImageId;
% Whether we should track only specified cars (If this option is set to
% true, only results corresponding to cars whose IDs are stored in the
% carIDs variable are evaluated and displayed. Else, all cars that are not
% occluded/truncated are evaluated and their results are displayed.)
trackSpecificCars = true;
% ID(s) of the car to track
% carIds = [2];  % Uncomment if running this script independently

% % Create an array to store the predictions
% yawPreds = zeros(size(imageList));

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


%% Initialize the data structs

disp('Initializing data structs ...');

% For each image in the list
for idx = 1:length(imageList)
    
    % Generate the file path for the current image to be processed
    imgFile = fullfile(kittiImageDir, sprintf('%06d.png',imageList(idx)));
    % Load the image
    img = imread(imgFile);
    
    % Tracklet for the current frame (usually comprises of multiple
    % annotations, again referred to as tracklets)
    tracklet = tracklets{idx};
    
    for j = 1:length(tracklet)
        % Current tracklet (annotation corresponding to a detection)
        curTracklet = tracklet(j);
        if ~strcmp(curTracklet.type, 'Car') && ~strcmp(curTracklet.type, 'Van') %|| ~ismember(curTracklet.id,carIds)
            continue
        end
        
        % If we only have to track specific cars, perform a check if
        % the car id of the current tracklet is present in the list of
        % car ids to be tracked
        if ~ismember(curTracklet.id, carIds) && trackSpecificCars
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
        
        if truncated ~= 0 || occluded ~= 0
            continue
        end
        
        numDetections = numDetections + 1;
        
        % Create the data structure for the current detection
        curDataStruct.bbox = single(bbox);
        curDataStruct.fileName = imgFile;
        curDataStruct.labels = single(pascalClassIndex(class));
        curDataStruct.carId = curTracklet.id;
        % Image number in the sequence
        curDataStruct.imgNo = idx;
        % Detection number in the image
        curDataStruct.detNo = j;
        % Append the ground-truth azimuth (used in evaluation)
        curDataStruct.yawTrue = curTracklet.ry;
        
        % ----------------------
        % Get single view-depth
        % ----------------------
        
        % KITTI camera intrinsics
        K = [721.5, 0.0, 609.5; 0, 721.5, 172.8; 0, 0 ,1.0];
        % KITTI camera height (in meters)
        camHeight = 1.52;
        
        % Y-coordinate (image plane) of the point identified to be on the
        % ground (we assume that the bottom of the bbox lies on the ground)
        yGround = curTracklet.y2;
        % Center it (subtract principal point)
        yGround = yGround - K(2,3);
        
        % Predicted depth
        curDataStruct.depthPred = K(1,1)*camHeight/yGround;
        
        % Actual depth
        curDataStruct.depthActual = curTracklet.t(3);
        
        % ----------------------
        
        dataStructs{numDetections} = curDataStruct;
        
    end
    
    
end


%% Visualize viewpoint predictions and plot them

disp('Computing single-view depth predictions ...');

% Get ids of cars in the scene, and the number of frames for which the
% corresponding car was unoccluded. The second argument indicates that the
% struct passed as the first argument is same as the data struct passed to
% the CNN.
% carStats = getCarStats(dataStructs, 'dataStruct');

% Ids of cars whose viewpoints have to be displayed
carIdsToShow = carIds(1);
trackSpecificCars = true;

% Whether or not to plot
shouldPlot = false;

% Initialize the plot
if shouldPlot
    figure(1);
end

% Variables to store the expected (true) and predicted azimuth values
expectedDepth = [];
predictedDepth = [];

% For each car detected
for idx = 1:length(dataStructs);
    
    curDataStruct = dataStructs{idx};
    % Continue, if the current detection doesn't correspond to a car that
    % we wish to track
    if ~ismember(curDataStruct.carId, carIdsToShow) && trackSpecificCars
        continue;
    end
    
    % Read in the image and display it
    img = imread(curDataStruct.fileName); 
    if shouldPlot
        imshow(img);
        hold on;
        % Plot the bbox for the current detection
        bbox = curDataStruct.bbox;
        bboxPlot = [bbox(1), bbox(2), bbox(3)-bbox(1), bbox(4)-bbox(2)];
        rectangle('Position', bboxPlot, 'LineWidth', 3, 'EdgeColor', 'g');
    end
    
    % True depth
    expectedDepth = [expectedDepth, curDataStruct.depthActual];
    % Predicted depth
    predictedDepth = [predictedDepth, curDataStruct.depthPred];
    
    if shouldPlot
        label_text = sprintf('%d', curDataStruct.carId);
        x = double((bbox(1) + bbox(3))/2);
        y = double(bbox(2));
        text(x, max(y-5,40), label_text, 'color', 'g', 'BackgroundColor', 'k', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold', 'FontSize', 8);
        
        pause(0.1);
    end
    
end

if shouldPlot
    hold off;
end


%% Get single-view depth statistics

% Generate a struct that is to be passed to the function to generate
% viewpoint consistency statistics. The struct contains general
% information viz., sequence number, car ID being tracked, start and end
% frames of the sequence
infoStruct.seqNum = sequenceNum;
infoStruct.carId = carIdsToShow;
infoStruct.firstFrameId = startImageId;
infoStruct.lastFrameId = endImageId;

% Whether or not to save the generated plots
saveDepthPlots = true;

% Get statistics. Optionally save them to the appropriate directory in the
% results directory (in 'cachedir'). The fourth parameter specifies whether
% or not to save these plots
getSingleViewDepthStats(expectedDepth, predictedDepth, infoStruct, saveDepthPlots);
