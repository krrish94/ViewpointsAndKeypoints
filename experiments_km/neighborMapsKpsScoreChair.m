function [mapsNeighbors] = neighborMapsKpsScoreChair(R, bbox, trainData)
% NEIGHBORMAPSKPSSCORE  Takes in the pose prediction for a test image, in
% the form of a roation matrix, the bounding box detection, and the train
% data. For the test image, NEIGHBORMAPSKPSSCORE gets a set of 'close'
% train images (with respect to viewpoint). It then uses the keypoint
% annotations of those images to initialize a prior heatmap of the keypoint
% locations, and smooths the heatmaps using a non-parametric mixture of
% gaussians (here, a distance transform based model).

% Inputs:
%   R: pose features extracted from the viewpoint estimation CNN, in the
%   form of a rotation matrix
%   bbox: bounding boxes of the training samples
%   trainingData: data structs containing information about the training
%   data


% Turn warnings off
warning off;

% Declaring global variables
globals;

%% Setting parameters

% Get the height and width of the heatmap
H = params.heatMapDims(2);
W = params.heatMapDims(1);

% Get the number of keypoints for the current class
% Kp = size(trainData(1).kps,1);
Kp = 10;

% Get the width, height, and aspect ratio of the box
wBox = bbox(3)-bbox(1);
hBox = bbox(4)-bbox(2);
bboxRatio = (wBox)/(hBox);

% Number of neighbors of the current keypoint
countNeighbors = zeros(Kp,1);
% Heatmaps of neighbors of keypoints
mapsNeighbors = zeros(H,W,Kp);

% Setting some parameters (???). Guessing these must be for the mexfile by
% Ross Girshick, implementing the bounded distance transform.
dtVal = 1/3;
dtRange = max(H,W)/2;

if(bboxRatio <1)
    dtValx=dtVal*(bboxRatio)^2;
    dtValy = dtVal;
else
    dtValx = dtVal;
    dtValy=dtVal/(bboxRatio^2);
end


% Threshold distance for a training sample to be called a neighbor. This is
% taken to be the (absolute) difference in their azimuths (in degrees).
thresh = 20;
% Number of neighbors of the current test sample
numNeighbors = 0;


%% Computing heatmaps

% For each training sample
for i = 1:length(trainData)
    % Compute the difference in the predicted pose of the test sample and
    % the current training sample. If that is less than the threshold
    if(norm(logm(R'*trainData(i).rot), 'fro') * 180/pi <= thresh)
        % Increment the number of neighbors of the current test sample
        numNeighbors = numNeighbors+1;
        % Get the bounding box
        bbox = trainData(i).bbox;
        % Get keypoint annotations
        kps = trainData(i).kps';
        % Find indices of keypoints that are not NaNs
        goodInds = find(~isnan(kps(1,:)));
        % Set bounds on keypoint locations, according to the bbox
        kps(1,:) = ceil(W*(kps(1,:)-bbox(1))/(bbox(3)-bbox(1)));
        kps(1,:) = max(kps(1,:),1);kps(1,:) = min(kps(1,:),W);
        kps(2,:) = ceil(H*(kps(2,:)-bbox(2))/(bbox(4)-bbox(2)));
        kps(2,:) = max(kps(2,:),1);kps(2,:) = min(kps(2,:),H);
        
        % For keypoints that are not NaN, i.e., for keypoints that are
        % visible in the current training image
        for kp = goodInds
            % Increment the neighbor count
            countNeighbors(kp) = countNeighbors(kp)+1;
            % Initialize the heatmap
            mapTmp = zeros(H,W);
            % Set the response at the locations of the keypoint annotations
            % to the maximum value (1.0)
            mapTmp(kps(2,kp),kps(1,kp)) = 1;
            % Compute a bounded distance transform of the current heatmap,
            % to make it smoother (uses Ross Girshick's fast_bounded_dt mex
            % function stored in the utils directory)
            [mapTmp,~,~] = fast_bounded_dt(mapTmp,dtValx,0,dtValy,0,dtRange);
            % Add this heatmap to mapsNeighbors
            mapsNeighbors(:,:,kp) = mapsNeighbors(:,:,kp) + mapTmp;
        end
    end
end


%% Normalizing heatmaps

% For each keypoint
for kp = 1:Kp
    % If there are no neighbors, set the heatmap to zeros (no prior)
    if(countNeighbors(kp)==0)
        mapsNeighbors(:,:,kp) = zeros(H,W);
    % Else, normalize the computed heatmaps by dividing the score at each
    % location by the number of neighbors used in the score computation for
    % that particular keypoint.
    else
        mapsNeighbors(:,:,kp) = mapsNeighbors(:,:,kp)/countNeighbors(kp);
        mapsNeighbors(:,:,kp) = mapsNeighbors(:,:,kp).^min(10,countNeighbors(kp)/10);
    end
end

%disp(numNeighbors/length(trainData));
%if(numNeighbors < 0.1*length(trainData))
%if(numNeighbors == 0)
    %mapsNeighbors = ones(size(mapsNeighbors));
%end

% Concatenate the heatmaps for all keypoints into a vector
mapsNeighbors = (mapsNeighbors(:))';

end

