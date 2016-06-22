function [] = collectViewpointData()
% COLLECTVIEWPOINTDATA  Collects viewpoint data for all 'car' images from
% the Imagenet and Pascal datasets and stores them as mat files in the 
% viewpointData, located in cachedir
% 
% For each image, COLLECTVIEWPOINTDATA saves a mat file containing the
% following variables
%       classIndex, overlap, bbox, regionIndex, eulers


% Declaring global variables
globals;

% Optionally, delete all mat folders previously created
delete([viewpointDataDir '/*.mat']);

% We're working with the car class
class = pascalIndexToClassLabel(params.classInds);

% Get rotation data for the class over Pascal
disp('Getting rotation data from Pascal');
rotationDataPascal = load(fullfile(cachedir, 'rotationDataPascal', class));
rotationDataPascal = rotationDataPascal.rotationData;
% Get only indices that have rotation data corresponding to Imagenet images
rotationDataPascal = rotationDataPascal(ismember({rotationDataPascal(:).dataset}, 'pascal'));

% Get rotation data for the class over Imagenet
disp('Getting rotation data from Imagenet');
rotationDataImagenet = load(fullfile(cachedir, 'rotationDataImagenet', class));
rotationDataImagenet = rotationDataImagenet.rotationData;
% Get only indices that have rotation data corresponding to Imagenet images
rotationDataImagenet = rotationDataImagenet(ismember({rotationDataImagenet(:).dataset}, 'imagenet'));

% Concatenate all the rotation data
rotationData = [rotationDataPascal rotationDataImagenet];

disp([num2str(length(rotationData)) ' detections obtained']);

% Variable to keep track of the number of files saved
numFilesSaved = 0;
% Variable to keep track of the number of training samples created
numSamples = 0;

% For each rotation data struct
for n = 1:length(rotationData)
    % Name of the data file to be created
    rcnnDataFile = fullfile(viewpointDataDir, [rotationData(n).voc_image_id '.mat']);
    % Get a bunch of overlapping boxes for the current detection
    bbox = overlappingBoxes(rotationData(n).bbox, rotationData(n).imsize);
    % Number of non-overlapping boxes thus formed
    nCands = size(bbox,1);
    
    numSamples = numSamples + nCands;
    
     % Initialize a vector containing class indices
    classIndex = params.classInds*ones(nCands,1);
    % Not really used
    overlap = ones(nCands,1);
    regionIndex = zeros(nCands,1);
    % Replicate the rotation data for all candidate overlapping boxes
    euler = repmat(rotationData(n).euler',nCands,1);
    imSize = rotationData(n).imsize;
    
    % Saving mat file
    dataset = '';
    if(~isempty(classIndex))
        if(exist(rcnnDataFile,'file'))
            rcnnData = load(rcnnDataFile);
            overlap = vertcat(rcnnData.overlap,overlap);
            euler = vertcat(rcnnData.euler,euler);
            bbox = vertcat(rcnnData.bbox,bbox);
            classIndex = vertcat(rcnnData.classIndex,classIndex);
            regionIndex = vertcat(rcnnData.regionIndex,regionIndex);
        end
        if n <= length(rotationDataPascal)
            dataset = 'pascal';
        else
            dataset = 'imagenet';
        end
        save(rcnnDataFile,'overlap','euler','bbox','classIndex','regionIndex','imSize', 'dataset');
        numFilesSaved = numFilesSaved + 1;
    end
end

disp([num2str(numFilesSaved) ' files saved']);
disp([num2str(numSamples) ' samples created']);

end
