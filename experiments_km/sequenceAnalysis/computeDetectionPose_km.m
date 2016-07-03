function pose = computeDetectionPose_km(feat)
% COMPUTEDETECTIONPOSES  Computes the azimuth, given the feature vector.
% Feature vectors must be of dimension N x 84, where N is the number of
% detections (feature vectors) and 84 is the dimensionality of each feature
% vector.


% Declaring global variables
globals;

% Number of bins used to quantize each angle
numBins = 21;

% Indices of cyclo-rotation, elevation, and azimuth in the feature vector
thetaInds = 1:21;
elevationInds = 22:42;
azimuthInds = 43:63;

% Number of feature vectors passed
N = size(feat, 1);

% Array to store outputs (Euler angles)
pose = zeros(N,3);

% For each feature vector
for i = 1:N
    % Current feature vector
    curFeat = feat(i,:);
    
    % Get the azimuth feature vector
    predFeatAz = curFeat(azimuthInds);
    % Predict the bin number corresponding to the azimuth
    [~, predAz] = max(predFeatAz, [], 2);
    
    % Get the elevation feature vector
    predFeatEl = curFeat(elevationInds);
    % Predict the bin number corresponding to the elevation
    [~, predEl] = max(predFeatEl, [], 2);
    
    % Get the cyclo-rotation feature vector
    predFeatTh = curFeat(thetaInds);
    [~, predTh] = max(predFeatTh, [], 2);
    
    pose(i,:) = [(predTh-11), (predEl-11), (predAz-0.5)]*360/numBins;
    pose(i,:) = [predTh, predEl, predAz];
end

end