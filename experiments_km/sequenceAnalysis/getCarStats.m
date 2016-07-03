function carStats = getCarStats(inStruct, structType)
% GETCARSTATS  Get ids of cars in the scene, and the number of frames for 
% which the corresponding car was unoccluded. The second argument indicates
% whether the struct passed as the first argument is same as the dataStruct 
% passed to the CNN, or if it is the tracklet struct obtained from the
% KITTI tracking development kit.


% Array to store carIds and the number of frames for which the car appears
% in the sequence.
carStats = [];

% If we're processing the datastruct created for the viewpoint CNN.
if strcmp(structType, 'dataStruct')
    % For each detection
    for i = 1:length(inStruct)
        % If it is an already seen car (and if it isn't the first car)
        if size(carStats,1) ~= 0 && ismember(inStruct{i}.carId, carStats(:,1))
            % Get the index in the 'carStats' matrix of the car id
            [~, idx] = ismember(inStruct{i}.carId, carStats(:,1));
            % Increment the number of frames in which the car was seen
            carStats(idx,2) = carStats(idx,2) + 1;
        % If the car has been seen for the first time
        else
            carStats(size(carStats,1)+1,1:2) = [inStruct{i}.carId, 1];
        end
    end
% If we're processing the tracklet struct from KITTI
elseif strcmp(structType, 'tracklet')
    for i = 1:length(inStruct)
        % Read the ith tracklet
        tracklet = inStruct{i};
        % For each detection in the current tracklet
        for j = 1:length(tracklet)
            
            curTracklet = tracklet(j);
            
            % Ignore the current tracklet if it doesn't correspond to a car
            % or van detection
            if ~strcmp(curTracklet.type, 'Car') && ~strcmp(curTracklet.type, 'Van')
                continue
            end
            
            % Determine whether or not the object is occluded
            occluded = curTracklet.occlusion;
            % Determine whether or not the object is truncated
            truncated = curTracklet.truncation;           
            
            % Ignore the current tracklet if the detection is occluded or
            % truncated
            if truncated ~= 0 || occluded ~= 0
                continue
            end
            
            % If it is an already seen car (and if it isn't the first car)
            if size(carStats,1) ~= 0 && ismember(curTracklet.id, carStats(:,1))
                % Get the index in the 'carStats' matrix of the car id
                [~, idx] = ismember(curTracklet.id, carStats(:,1));
                % Increment the number of frames in which the car was seen
                carStats(idx,2) = carStats(idx,2) + 1;
                % If the car has been seen for the first time
            else
                carStats(size(carStats,1)+1,1:2) = [curTracklet.id, 1];
            end
            
        end
    end
else
    carStats = [];
end

end