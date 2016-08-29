function seqAndCarIds = getSequencesWithLongTracks(minFrames)
% GETSEQUENCESWITHLONGTRACKS  Processes KITTI data and retrieves sequences
% and ids of cars where the car was visible for more than a minimum number
% of frames specified.


%% Preprocessing

% Declaring global variables
globals;

% If the minimum number of frames isn't specified, retrieve cars that were
% visible for more than 100 frames, and look in all sequences.
if nargin == 0
    minFrames = 100;
    startSequence = 0;
    endSequence = 10;
    % Sequences to search through
    sequences = startSequence:endSequence;
elseif nargin == 1
    
    startSequence = 1;
    endSequence = 1;
    
    % % Sequences to search through
    sequences = startSequence:endSequence;
    
    % Sequence 16 somehow doesn't work (maybe it's a pedestrian seq). So
    % use this instead.
    % sequences = [1:16, 18:20];
end


%% Initialize variables for KITTI data directories

% Base directory (containing KITTI data)
kittiBaseDir = fullfile(basedir, 'data', 'KITTI');
% Directory containing KITTI labels (for training sequences)
kittiLabelDir = fullfile(kittiBaseDir, 'label_02');
% Directory containing camera calibration data
kittiCalibDir = fullfile(kittiBaseDir, 'calib');


%% Get sequence information

% Initialize array to hold sequence number and their corresponding car ids
seqAndCarIds = [];

% For each sequence specified
for seqIdx = 1:length(sequences)
    
    % Root directory containing KITTI images (for training sequences)
    kittiImageDir = fullfile(kittiBaseDir, sprintf('image_02/%04d', sequences(seqIdx)));
    
    % Get number of images in the sequence
    numFrames = length(dir(fullfile(kittiImageDir)))-2;
    
    % ID of the first image to process (in the sequence specified)
    startImageId = 0;
    % ID of the last image to process (in the sequence specified)
    % endImageId = 100;
    endImageId = numFrames-1;
    
    % Get tracklets corresponding to the current sequence
    tracklets = readLabels(kittiLabelDir, sequences(seqIdx));
    % Get stats of the current sequence. 'carStats' is an N x 2 matrix, where
    % each row is of the form [carId, numFrames].
    carStats = getCarStats(tracklets, 'tracklet');
    
    % Get the ids of the cars that are visible for > minFrames
    cars = find(carStats(:,2) > minFrames);
    % Add the cars, along with the sequence number and the number of frames
    % that they appear for, to the output matrix
    for j = 1:size(cars,1)
        
        if size(seqAndCarIds,1) == 0
            seqAndCarIds = [sequences(seqIdx), carStats(cars(j),1), carStats(cars(j),2)];
        else
            seqAndCarIds(end+1,:) = [sequences(seqIdx), carStats(cars(j),1), carStats(cars(j),2)];
        end
    end
    
end


end