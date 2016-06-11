function [] = readKpsData(class)
%READKPSDATA Summary of this function goes here
%   Detailed explanation goes here

% Declaring global variables
globals;
% Get class index
classInd = pascalClassIndex(class);

% Get the names of all mat files in the annotation directory
fnames = getFileNamesFromDirectory(annotationDir,'types',{'.mat'});
% Load the train/test split of Pascal
load(fullfile(cachedir,'pascalTrainValIds.mat'));

% Initialize a data struct to hold the image id, bbox, kps, etc.
dataStruct.voc_image_id = {};
dataStruct.bbox = [];
dataStruct.kps = {};
dataStruct.voc_rec_id = [];
dataStruct.occluded = [];

% For each annotation present in the list of files
for i=1:length(fnames)
    % If it is not a member of either train or val, ignore it
    if(~ismember(fnames{i}(1:end-4),vertcat(valIds,trainIds)))
        continue;
    end
    % Load the ground truth annotation file
    gt = load(fullfile(annotationDir,fnames{i}));
    % Check if kps is empty
    goodInds = ismember(gt.class,classInd) & cellfun(@(x)~isempty(x),gt.kps');
    % Exclude the occluded images
    if(params.excludeOccluded)
        goodInds = goodInds & ~gt.occluded & ~gt.truncated & ~gt.difficult ;
    else
        goodInds = goodInds & ~gt.difficult ;
    end
    %goodInds = ismember(gt.class,classInds);
    
    % If the current annotation is good-to-go
    if(sum(goodInds))
        % Populate the data struct
        dataStruct.voc_image_id = vertcat(dataStruct.voc_image_id,repmat({fnames{i}(1:end-4)},sum(goodInds),1));
        dataStruct.voc_rec_id = vertcat(dataStruct.voc_rec_id,gt.voc_rec_id(goodInds));
        dataStruct.kps = horzcat(dataStruct.kps,gt.kps(goodInds));
        dataStruct.bbox = vertcat(dataStruct.bbox,gt.bbox(goodInds,:));
        dataStruct.occluded = vertcat(dataStruct.occluded,gt.occluded(goodInds) | gt.truncated(goodInds));
    end
end

% Save the keypoint data to disk
fname = fullfile(kpsPascalDataDir,class);
save(fname,'dataStruct');

end