function [rotationData] = augmentKps(rotationData,dataStruct)
%AUGMENTKPS  Takes as input rotation data (for Pascal) and a dataStruct
%   containing information about all the detections. For each detection,
%   i.e., for each dataStruct object, AUGMENTKPS assigns ground-truth
%   keypoint annotations into the field dataStruct.kps by comparing the
%   bboxes present in dataStruct with those present in the ground-truth.


% Declaring global variables
globals;

% Number of keypoints for the class of the current training sample
nKp = size(dataStruct.kps{1},1);

% For each member of the training set
for i=1:length(rotationData)
    % Find the index (indices) of the current image in the rotation data
    js = find(ismember(dataStruct.voc_image_id,{rotationData(i).voc_image_id}));
    % Get the bounding box
    boxThis = rotationData(i).bbox;
    % If the image is not present in the train set, set all keypoints to
    % NaN
    if(isempty(js))
        rotationData(i).kps = nan(nKp,2);
    else
        % Get the bounding boxes
        boxes = dataStruct.bbox(js,:);
        % Get the distance of all the bounding boxes in the image from the
        % bbox present in the current dataStruct being processed
        boxDist = sum(abs(bsxfun(@minus,boxes,boxThis)),2);
        % Get the indices of bounding boxes which are very close, i.e.,
        % whose boxDist is very small (I'm assuming this is done to
        % accomodate noise in bounding box prediction).
        jThis = js(boxDist <= 4);
        % If no such index exists, set the keypoint locations to NaN
        if(isempty(jThis))
            rotationData(i).kps = nan(nKp,2);
        % Else, initialize the set of keypoints for the current bbox to
        % that of the first candidate bounding box.
        else
            rotationData(i).kps = dataStruct.kps{jThis(1)};
        end
    end

end

end