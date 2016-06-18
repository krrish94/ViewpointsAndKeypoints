function [] = vpsPascalDataCollect()

%RCNNIMAGENETDATACOLLECT Summary of this function goes here
%   Detailed explanation goes here


% Declaring global variables
globals;

% We're working with the car class
class = pascalIndexToClassLabel(params.classInds);

% Get rotation data for the class
rotationData = load(fullfile(cachedir, 'rotationDataPascal', class));
rotationData = rotationData.rotationData;
% Get only indices that have rotation data corresponding to Pascal images
rotationData = rotationData(ismember({rotationData(:).dataset}, 'pascal'));

% Optionally, delete all mat folders previously created
% delete([rcnnVpsPascalDataDir '/*.mat']);

% For each rotation data struct
for n = 1:length(rotationData)
    % Name of the data file to be created
    rcnnDataFile = fullfile([rcnnVpsPascalDataDir '_test'], [rotationData(n).voc_image_id '.mat']);
    % Get a bunch of overlapping boxes for the current detection
    bbox = overlappingBoxes(rotationData(n).bbox, rotationData(n).imsize);
    % Number of non-overlapping boxes thus formed
    nCands = size(bbox,1);
    
    % Initialize a vector containing class indices
    classIndex = params.classInds*ones(nCands,1);
    % Not really used
    overlap = ones(nCands,1);
    regionIndex = zeros(nCands,1);
    % Replicate the rotation data for all candidate overlapping obxes
    euler = repmat(rotationData(n).euler', nCands, 1);
    imSize = rotationData(n).imsize;
    
    % Saving file
    if(~isempty(classIndex))
        if(exist(rcnnDataFile,'file'))
            rcnnData = load(rcnnDataFile);
            overlap = vertcat(rcnnData.overlap,overlap);
            euler = vertcat(rcnnData.euler,euler);
            bbox = vertcat(rcnnData.bbox,bbox);
            classIndex = vertcat(rcnnData.classIndex,classIndex);
            regionIndex = vertcat(rcnnData.regionIndex,regionIndex);
        end
        save(rcnnDataFile,'overlap','euler','bbox','classIndex','regionIndex','imSize');
    end
end

end
