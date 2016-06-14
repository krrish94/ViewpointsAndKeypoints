%% Predict keypoint locations and plot them

% Note: Run extractKpsKITTI.m before running this script

% For each detection in the image set
for idx = 1:length(dataStructs)
    
    % Read in the image
    im = imread(fullfile(dataStructs{idx}.fileName));
    testFeat = featAll(idx,:);
    bbox = dataStructs{idx}.bbox;
    
    [kpCoords,scores] = maxLocationPredict(testFeat, bbox, params.heatMapDims);
    kpCoords = kpCoords(1:2, 1:14);
    [b, bi] = sort(scores, 'descend');
    [~, ind] = find(scores >= 0.8);
    kpCoords = kpCoords(1:2, ind);
    
    bbox2(1) = bbox(1); bbox2(2) = bbox(2); bbox2(3) = bbox(3)-bbox(1); bbox2(4) = bbox(4)-bbox(2);
    imshow(im);
    hold on
    scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled')
    % scatter(kps(:,1),kps(:,2),50,'b','filled')
    rectangle('Position', bbox2, 'LineWidth', 3, 'EdgeColor', 'g');
    hold off
    
    pause;
end