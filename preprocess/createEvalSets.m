function [] = createEvalSets(class)


disp('Creating Eval Sets');

% Declare global variables
globals;

% Location for saving the evaluation file
mkdirOptional(fullfile(cachedir,'evalSets'));
saveFile = fullfile(cachedir,'evalSets',class);

%% Load data for train and test sets

% Different load directories for Imagenet and Pascal
if strcmp(params.vpsDataset, 'imagenet')   
    % Load the features for the class from the mat file
    var = load(fullfile(cachedir, 'rcnnPredsVps',[params.features '_imagenet'], class));
    feat = var.feat;
    % Load rotation data for the corresponding class
    var = load(fullfile(rotationImagenetDataDir, class));
    rotData = var.rotationData;
elseif strcmp(params.vpsDataset, 'pascal') 
    % Load the features for the class from the mat file
    var = load(fullfile(cachedir,'rcnnPredsVps',params.features,class));
    feat = var.feat;
    % Load rotation data for the corresponding class
    var = load(fullfile(rotationPascalDataDir,class));
    rotData = var.rotationData;
end


% Parse rotation information
rec_ids = vertcat(rotData.voc_rec_id);
bboxes = vertcat(rotData.bbox);
occluded = vertcat(rotData.occluded);
truncated = vertcat(rotData.truncated);
IoUs = vertcat(rotData.IoU);
%voc_ids = vertcat(rotData.voc_image_id);
objectInds = vertcat(rotData.objectInd);

%eulers = horzcat(rotData.euler);eulers = eulers';

% Initializing variables to store a few parameters
eulers = [];rots = [];goodInds = [];voc_ids = {};masks = {};
% For the rotation data from each detection
for i=1:length(rotData)
    % If Euler data is present and if not all Euler angles are zeros
    if(~isempty(rotData(i).euler) && sum(rotData(i).euler == 0)~=3)
        % Append this index to the list of good indices
        goodInds(end+1) = i;
        % Append the rotation matrix (in rotData(i).rot(:)) to rots, i.e.,
        % stack the rotation matrix into a vector and store it
        rot = rotData(i).rot(:);
        rots(end+1,:)=rot';
        % Append the Euler angles of the rotation data to 'eulers'
        eulers(end+1,:) = rotData(i).euler';
        % Append the image id to 'voc_ids'
        voc_ids{end+1} = rotData(i).voc_image_id;
        % Append mask data to 'masks'
        masks{end+1} = rotData(i).mask;
    end
end

% Get only features for images where Euler data is present and not all
% Euler angles are zeros. Call such indices 'good indices'.
feat = feat(goodInds,:);
% Get the rectangle ids for good indices
rec_ids = rec_ids(goodInds);
%voc_ids = voc_ids(goodInds);
% Get the bounding boxes for good indices
bboxes = bboxes(goodInds,:);
% Get the occlusion state of good indices
occluded = occluded(goodInds);
% Get the truncation state of good indices
truncated = truncated(goodInds);
% IoU = ? (TODO: look up IoU)
IoUs = IoUs(goodInds);
% Get the object indices for good indices
objectInds = objectInds(goodInds);


%% Creating training and validation partitions

if strcmp(params.vpsDataset, 'imagenet')
    % Get the train and validation indices for PASCAL VOC
    sets = load(fullfile(cachedir,'imagenetTrainIds'));
elseif strcmp(params.vpsDataset, 'pascal')
    % Get the train and validation indices for PASCAL VOC
    sets = load(fullfile(cachedir,'pascalTrainValIds'));
end

% Training Set

if strcmp(params.vpsDataset, 'imagenet')
    % Create a list of all images for which rotation data is available
    
    % Get names of files for which annotations can be used
    goodIndNames = {};
    for i = 1:length(goodInds)
        goodIndNames{i} = rotData(i).voc_image_id;
    end
    % Split them into train and test sets
    
    
    inds = ismember(voc_ids,sets.fnamesTrain);
elseif strcmp(params.vpsDataset, 'pascal')
    % Get indices that are part of trainIds
    inds = ismember(voc_ids,sets.trainIds);
end

% Get data parameters of such indices
data.feat = feat(inds,:);
data.eulers = eulers(inds,:);
data.rots = rots(inds,:);
data.voc_ids = voc_ids(inds);
data.rec_ids = rec_ids(inds);
data.bboxes = bboxes(inds,:);
data.occluded = occluded(inds,:);
data.truncated = truncated(inds,:);
data.IoUs = IoUs(inds,:);
data.masks = masks(inds);
data.objectInds = objectInds(inds);
% Store them in the train set
train = data;

% Validation Set

if strcmp(params.vpsDataset, 'imagenet')
    valInds = setdiff(goodInds, inds);
    inds = valInds;
elseif strcmp(params.vpsDataset, 'pascal')
    % Get indices that are part of valIds
    inds = ismember(voc_ids,sets.valIds);
end

% Get data parameters of such indices
data.feat = feat(inds,:);
data.eulers = eulers(inds,:);
data.rots = rots(inds,:);
data.voc_ids = voc_ids(inds);
data.rec_ids = rec_ids(inds);
data.bboxes = bboxes(inds,:);
data.occluded = occluded(inds,:);
data.truncated = truncated(inds,:);
data.IoUs = IoUs(inds,:);
data.masks = masks(inds);
data.objectInds = objectInds(inds);
% Store them in the test set
test = data;


% Save the train and test sets
save(saveFile,'train','test','-v7.3');

end

