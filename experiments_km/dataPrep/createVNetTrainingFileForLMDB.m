function [] = createVNetTrainingFileForLMDB(fNameSuffix)
% CREATEVNETTRAININGFILEFORLMDB  Create a training file for the viewpoint 
%   network. fNameSuffix is the name to be given to the VNet train file.

% Format of the window file created
%   absolute/image/path bbox(x1,y1,x2,y2) sin(azimuth) cos(azimuth) ...
%                                     ... sin(elevation) cos(elevation)


% Declaring global variables
globals;

% We're interested in the 'car' class
class = 'car';

% Load train ids for Pascal and Imagenet
load(fullfile(cachedir, 'imagenetTrainIds.mat'));
load(fullfile(cachedir, 'pascalTrainValIds.mat'));

% Load rotation data for Pascal and Imagenet
rotationData = load(fullfile(rotationJointDataDir, class));
rotationData = rotationData.rotationData;

% Get all voc_image_ids from rotation data
rData = {rotationData(:).voc_image_id};

% Array to store training indices
trainIndices = [];

% Retain only training data
disp('Gathering training data ...');
reverseStr = '';

for i = 1:length(rotationData)
    % Display progress (after every 100 iters)
    if mod(i,100) == 0
        percentDone = i*100/length(rotationData);
        msg = sprintf('Done: %3.1f', percentDone);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    % If the current voc_image_id exists in train ids, add it to train data
    if strcmp(rotationData(i).dataset, 'pascal')
        if strmatch(rData{i}, trainIds, 'exact')
            trainIndices = [trainIndices, i];
        end
    elseif strcmp(rotationData(i).dataset, 'imagenet')
        if strmatch(rData{i}, fnamesTrain, 'exact')
            trainIndices = [trainIndices, i];
        end
    end
end

% Create the train file
disp('Generating CNN window file');

% Open (create) the text file (for writing)
txtFile = fullfile(VNetTrainFilesDir, [fNameSuffix '.txt']);
fid = fopen(txtFile, 'w+');

count = 0;

% Variables to display progress text
reverseStr = '';

% Write data to file
for j = 1:length(trainIndices)
    
    % Display progress (after every 100 iters)
    if mod(j,100) == 0
        percentDone = j*100/length(trainIndices);
        msg = sprintf('Done: %3.1f', percentDone);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    % Current rotation data
    cur = rotationData(trainIndices(j));
    % Absolute file path
    absPath = '';
    if strcmp(cur.dataset, 'pascal')
        absPath = fullfile(pascalImagesDir, [cur.voc_image_id, '.jpg']);
    elseif strcmp(cur.dataset, 'imagenet')
        absPath = fullfile(imagenetImagesDir, [cur.voc_image_id, '.jpg']);
    end
    
    % Write data to file
    fprintf(fid, '%s %d %d %d %d %f %f %f %f \n', absPath, cur.bbox(1), ...
        cur.bbox(2), cur.bbox(3), cur.bbox(4), sin(cur.euler(3)), ...
        cos(cur.euler(3)), sin(cur.euler(1)), cos(cur.euler(1)));
    
end

fprintf('\n');

% Close the file
fclose(fid);

end
