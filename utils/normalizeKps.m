function [kpNum,kpCoords] = normalizeKps(kps,bbox,dims)
% NORMALIZEKPS  Normalizes keypoints according to the bounding box and
% heatmap dimensions, such that all coordinates are in the range [0,1].


% Scaling factor for rows
deltaX = (bbox(3)-bbox(1)+1)/dims(1);
% Scaling factor for cols
deltaY = (bbox(4)-bbox(2)+1)/dims(2);

% Number of keypoints
kpInds = 1:size(kps,1);
% Perform scaling such that all keypoint coordinates are in the range [0,1]
kps(:,1) = floor((kps(:,1)-bbox(1))/deltaX) + 1;
kps(:,2) = floor((kps(:,2)-bbox(2))/deltaY) + 1;

% Set all indices to 'good'. The gaussian used in a later section of the
% code will sort things out.
goodInds = true(size(kpInds));
kpNum = kpInds(goodInds);kpCoords = kps(goodInds,:);

end