
%% Note: Run extractKpsKITTI.m before running this script


%% Group the detections by image, for easy visualization

% Current index (of image in sequence)
curImgIdx = 0;
% Current index (of detection in image)
curDetIdx = 0;

% Filename corresponding to the previous image being processed
prevFileName = '';

% Cell to hold all detections
kpDetections = {};
% Array to hold detections only for the current image
curDetections = {};

% For each dataStruct
for idx = 1:length(dataStructs)
    % Compare the current image filename with the previous. If the filename
    % has changed, create a new index in the cell
    if ~strcmp(dataStructs{idx}.fileName, prevFileName)
        % Increment the index of the current image
        curImgIdx = curImgIdx + 1;
        % Reset the index of the detections in the current image
        curDetIdx = 1;
        % Store the data struct at the appropriate index
        kpDetections{curImgIdx}{curDetIdx} = dataStructs{idx};
        % Update the previous image filename
        prevFileName = dataStructs{idx}.fileName;
    else
        % Increment the index of the detections in the current image
        curDetIdx = curDetIdx + 1;
        % Store the data struct at the appropriate index
        kpDetections{curImgIdx}{curDetIdx} = dataStructs{idx};
    end
end

% % The following snippet has been abandoned after failing a unit test.
% % For each dataStruct
% for idx = 1:length(dataStructs)
%     % If the current image filename is different from the one that we've
%     % been processing until now
%     if ~strcmp(dataStructs{idx}.fileName, prevFileName)
%         
%         % If this is not the first index (idx), then we've completed
%         % collecting detections from one image. Add them to kpDetections.
%         if idx ~= 1
%             kpDetections{curImgIdx} = curDetections;
%         end
%         
%         % Update prevFileName
%         prevFileName = dataStructs{idx}.fileName;
%         % Update the index of the current image
%         curImgIdx = curImgIdx + 1;
%         % Update the detection index in the current image
%         curDetIdx = 1;
%         % Array to hold detections in the current image
%         curDetections = {};
%     else
%         % Update curDetIdx
%         curDetIdx = curDetIdx + 1;
%     end
%     % Append the current dataStruct to the appropriate array
%     curDetections{curDetIdx} = dataStructs{idx};
% end


%% Predict keypoint locations and plot them

% Whether or not to write video output
writeVideoOutput = false;

% Initialize the GUI
fig = figure(1);
imgFile = fullfile(kittiImageDir, sprintf('%06d.png', imageList(1)));
img = imread(imgFile);

% Create a folder to store the output images
if writeVideoOutput
    mkdir(sprintf('kp_results/seq%02d_%03d_%03d', sequenceNum, startImageId, endImageId));
    set(fig, 'PaperPositionMode', 'auto');
end

% Sum of all indices we've seen thus far
idxSum = 0;

% For each image
for idx = 1:length(kpDetections)
    
    curKpDetection = kpDetections{idx};
    
    % Read in the current image
    im = imread(fullfile(curKpDetection{1}.fileName));
    
    % Draw the image
    imshow(im);
    hold on;
    % Plot title
    text(size(im,2)/2,3,sprintf('Keypoint Demo'),'color','g','HorizontalAlignment','center','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    
    % Legend
    text(0,00,'Not occluded','color','g','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    text(0,30,'Partly occluded','color','y','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    text(0,60,'Fully occluded','color','r','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    text(0,90,'Unknown','color','w','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    text(0,120,'Don''t care region','color','c','HorizontalAlignment','left','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    
    % Frame number
    text(size(im,2),0,sprintf('Sequence %d frame %d (%d/%d)', sequenceNum, imageList(idx), idx, length(imageList)),'color','g','HorizontalAlignment','right','VerticalAlignment','top','FontSize',14,'FontWeight','bold','BackgroundColor','black');
    
    % Specifying colors for occlusion levels
    occlusionColors = {'g', 'y', 'r', 'w', 'c'};
    
    for j = 1:length(curKpDetection)
        
        % Increment idxSum
        idxSum = idxSum + 1;
        
        testFeat = featAll(idxSum, :);
        bbox = curKpDetection{j}.bbox;
        
        [kpCoords,scores] = maxLocationPredict(testFeat, bbox, params.heatMapDims);
        kpCoords = kpCoords(1:2, 1:14);
        % Uncomment the following line if you only want to retain some
        % 'confident' keypoints
%         [b, bi] = sort(scores, 'descend');
%         [~, ind] = find(scores >= 0.4);
%         kpCoords = kpCoords(1:2, ind);
        
        bbox2(1) = bbox(1); bbox2(2) = bbox(2); bbox2(3) = bbox(3)-bbox(1); bbox2(4) = bbox(4)-bbox(2);
        
        % Use this if you only want to see the keypoints (scatter plot)
        % scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled')
        
        
        % Use this if you want to see overlaid wireframes
        numKps = size(kpCoords, 2);
        % Generate distinguishable colors with respect to a white background
        colors = distinguishable_colors(numKps, [0, 0, 0]);
        % First, plot the keypoints (vertices of the wireframe)
        wireframe = double(kpCoords);
        scatter(wireframe(1,:), wireframe(2,:), repmat(20, 1, numKps), colors, 'filled');
        % Plot text labels for car keypoints, using distinguishable colors
        text(wireframe(1,1), wireframe(2,1), 'L\_F\_WheelCenter', 'color', colors(1,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,2), wireframe(2,2), 'R\_F\_WheelCenter', 'color', colors(2,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,3), wireframe(2,3), 'L\_B\_WheelCenter', 'color', colors(3,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,4), wireframe(2,4), 'R\_B\_WheelCenter', 'color', colors(4,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,5), wireframe(2,5), 'L\_HeadLight', 'color', colors(5,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,6), wireframe(2,6), 'R\_HeadLight', 'color', colors(6,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,7), wireframe(2,7), 'L\_TailLight', 'color', colors(7,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,8), wireframe(2,8), 'R\_TailLight', 'color', colors(8,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,9), wireframe(2,9), 'L\_SideViewMirror', 'color', colors(9,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,10), wireframe(2,10), 'R\_SideViewMirror', 'color', colors(10,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,11), wireframe(2,11), 'L\_F\_RoofTop', 'color', colors(11,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,12), wireframe(2,12), 'R\_F\_RoofTop', 'color', colors(12,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,13), wireframe(2,13), 'L\_B\_RoofTop', 'color', colors(13,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        text(wireframe(1,14), wireframe(2,14), 'R\_B\_RoofTop', 'color', colors(14,:), 'FontSize', 10, 'BackgroundColor', [0, 0, 0]);
        % L_F_RoofTop -> R_F_RoofTop -> R_B_RoofTop -> L_B_RoofTop
        edges = [11, 12; 12, 14; 14, 13; 13, 11;];
        % L_HeadLight -> R_HeadLight -> R_TailLight -> L_TailLight
        edges = [edges; 5, 6; 6, 8; 8, 7; 7, 5];
        % L_Headlight -> L_F_RoofTop
        edges = [edges; 5, 11];
        % R_HeadLight -> R_F_RoofTop
        edges = [edges; 6, 12];
        % L_TailLight -> L_B_RoofTop
        edges = [edges; 7, 13];
        % R_TailLight -> R_B_RoofTop
        edges = [edges; 8, 14];
        % L_F_WheelCenter -> R_F_WheelCenter -> R_B_WheelCenter -> L_B_WheelCenter
        edges = [edges; 1, 2; 2, 4; 4, 3; 3, 1];
        % L_HeadLight -> L_F_WheelCenter
        edges = [edges; 5, 1];
        % R_HeadLight -> R_F_WheelCenter
        edges = [edges; 6, 2];
        % L_TailLight -> L_B_WheelCenter
        edges = [edges; 7, 3];
        % R_TailLight -> R_B_WheelCenter
        edges = [edges; 8, 4];
        % L_SideViewMirror -> L_HeadLight
        edges = [edges; 9, 5];
        % R_SideViewMirror -> R_HeadLight
        edges = [edges; 10, 6];
        % Generate distinguishable colors (equal to the number of edges). The
        % second parameter to the function is the background color.
        colors = distinguishable_colors(size(edges,1), [1, 1, 1]);
        
        % Draw each edge in the plot
        for i = 1:size(edges, 1)
            plot(wireframe(1,[edges(i,1), edges(i,2)]), wireframe(2, [edges(i,1), edges(i,2)]), ...
                'LineWidth', 2, 'Color', colors(i,:));
        end
        
        
        rectangle('Position', bbox2, 'LineWidth', 3, 'EdgeColor', 'g');
        
    end
    
    if writeVideoOutput
        print(sprintf('kp_results/seq%02d_%03d_%03d/%03d.png', sequenceNum, startImageId, endImageId, imageList(idx)), '-dpng', '-r0');
    end
    
    hold off;
    pause(0.1);
    
end



%% Old (primitive) GUI

% % For each detection in the image set
% for idx = 1:length(dataStructs)
%     
%     % Read in the image
%     im = imread(fullfile(dataStructs{idx}.fileName));
%     testFeat = featAll(idx,:);
%     bbox = dataStructs{idx}.bbox;
%     
%     [kpCoords,scores] = maxLocationPredict(testFeat, bbox, params.heatMapDims);
%     kpCoords = kpCoords(1:2, 1:14);
%     [b, bi] = sort(scores, 'descend');
%     [~, ind] = find(scores >= 0.8);
%     kpCoords = kpCoords(1:2, ind);
%     
%     bbox2(1) = bbox(1); bbox2(2) = bbox(2); bbox2(3) = bbox(3)-bbox(1); bbox2(4) = bbox(4)-bbox(2);
%     imshow(im);
%     hold on
%     scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled')
%     % scatter(kps(:,1),kps(:,2),50,'b','filled')
%     rectangle('Position', bbox2, 'LineWidth', 3, 'EdgeColor', 'g');
%     hold off
%     
%     pause;
% end