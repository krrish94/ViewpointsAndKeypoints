%% Displays ground truth pose (yaw/azimuth) for a set of KITTI images

% We're intereseted only in the 'car' class
class = 'car';

% Turn off Matlab warnings
warning('off', 'all');

% Declare global variables
globals;

% % Initialize the network for viewpoint prediction
% initViewpointNet;

% Add KITTI's Matlab code directory to path (the visualization one)
addpath /home/km/ViewpointsAndKeypoints/data/KITTI/devkit_tracking/matlab/

% Whether or not to write output to a video file
writeVideoOutput = false;

% Whether to show coarse (left, right, front, rear) or fine (angle in
% degrees) labels over the selected bounding boxes
showCoarseLabels = false;


%% Parameters for KITTI (test data)

% ID of the sequence to be processed
sequenceNum = 1;

% Mode ('manual', or 'auto'). Specifies if the user will input the bounding
% box or if they have to be picked up from the ground truth.
bboxMode = 'auto';


% Base directory (containing KITTI data)
kittiBaseDir = fullfile(basedir, 'data', 'KITTI');
% Root directory containing KITTI images (for training sequences)
kittiImageDir = fullfile(kittiBaseDir, sprintf('image_02/%04d', sequenceNum));
% Directory containing KITTI labels (for training sequences)
kittiLabelDir = fullfile(kittiBaseDir, 'label_02');
% Directory containing camera calibration data
kittiCalibDir = fullfile(kittiBaseDir, 'calib');

% Get number of images in the sequence
numFrames = length(dir(fullfile(kittiImageDir)))-2;

% ID of the first image to process (in the sequence specified)
startImageId = 27;
% ID of the last image to process (in the sequence specified)
% endImageId = 0;
endImageId = numFrames-1;
% Creating a list of images to process
imageList = startImageId:endImageId;
% Whether we should track only specified cars (If this option is set to
% true, only results corresponding to cars whose IDs are stored in the
% carIDs variable are evaluated and displayed. Else, all cars that are not
% occluded/truncated are evaluated and their results are displayed.)
trackSpecificCars = false;
% ID(s) of the car to track
carIds = [1];

% % Create an array to store the predictions
% yawPreds = zeros(size(imageList));

% Get calibration parameters
% parameters: calib directory, sequence num, camera id (here, 2)
P = readCalibration(kittiCalibDir, sequenceNum, 2);

% Load labels for the current sequence
tracklets = readLabels(kittiLabelDir, sequenceNum);

% Initialize the GUI
fig = figure(1);
imgFile = fullfile(kittiImageDir, sprintf('%06d.png', imageList(1)));
img = imread(imgFile);

% Create a folder to store the output images
if writeVideoOutput
    mkdir(sprintf('vp_results/seq%02d_%03d_%03d', sequenceNum, startImageId, endImageId));
    set(fig, 'PaperPositionMode', 'auto');
end

% Predict pose for each image
for i = 1:length(imageList)
    
    % Generate the file path for the current image to be processed
    imgFile = fullfile(kittiImageDir, sprintf('%06d.png',imageList(i)));
    % Load the image
    img = imread(imgFile);
    
    if strcmp(bboxMode, 'manual')
        % Display the image, and wait for the user to draw a bounding box
        % around the object of interest
        imshow(img);
        r = imrect;
        position = wait(r);
        bbox = single([position(1), position(2), position(1)+position(3), position(2)+position(4)]);
    else
        
        % Draw the image
        imshow(img);
        hold on;
        % Plot title
        text(size(img,2)/2,3,sprintf('KITTI Pose'),'color','g','HorizontalAlignment','center','VerticalAlignment','top','FontSize',12,'FontWeight','bold','BackgroundColor','black');
       
        % Frame number
        text(size(img,2),0,sprintf('Sequence %d frame %d (%d/%d)', sequenceNum, imageList(i), i, length(imageList)),'color','g','HorizontalAlignment','right','VerticalAlignment','top','FontSize',12,'FontWeight','bold','BackgroundColor','black');
        
        % Specifying colors for occlusion levels
        occlusionColors = {'g', 'y', 'r', 'w', 'c'};
        
        % Tracklet for the current frame (usually comprises of multiple
        % annotations, again referred to as tracklets)
        tracklet = tracklets{imageList(i)+1};
        
        for j = 1:length(tracklet)
            % Current tracklet (annotation corresponding to a detection)
            curTracklet = tracklet(j);
            if ~strcmp(curTracklet.type, 'Car') && ~strcmp(curTracklet.type, 'Van') %|| ~ismember(curTracklet.id,carIds)
                continue
            end
            
            % If we only have to track specific cars, perform a check if
            % the car id of the current tracklet is present in the list of
            % car ids to be tracked
            if ~ismember(curTracklet.id, carIds) && trackSpecificCars
                continue
            end
            
            % Get the bounding box (x1,y1,x2,y2), required by the CNN)
            bbox = single([curTracklet.x1, curTracklet.y1, curTracklet.x2, curTracklet.y2]);
            % Get the bounding box (x,y,w,h), required by the rect command
            bboxPos = int16([curTracklet.x1, curTracklet.y1, (curTracklet.x2-curTracklet.x1+1), (curTracklet.y2-curTracklet.y1+1)]);
            % Determine whether or not the object is occluded
            occluded = curTracklet.occlusion;
            % Determine whether or not the object is truncated
            truncated = curTracklet.truncation;
            % Draw the rectangle
            rectangle('Position', bboxPos, 'EdgeColor', occlusionColors{occluded+1}, 'LineWidth', 3);
            
            % Create the data structure for the current detection
            dataStruct.bbox = bbox;
            dataStruct.fileName = imgFile;
            dataStruct.labels = single(pascalClassIndex(class));
            
            % Get ground truth yaw
            yaw_true = curTracklet.ry*180/pi;
            yaw = yaw_true;
            
            % Draw the label above the object
            
            if showCoarseLabels
                % Coarse label text
                if (yaw >= 18 && yaw <= 21) || (yaw >= 1 && yaw <= 3)
                    label_text = 'front';
                elseif (yaw >=4 && yaw <= 9)
                    label_text = 'left';
                elseif (yaw >= 10 && yaw <= 12)
                    label_text = 'rear';
                elseif(yaw >= 13 && yaw <= 17)
                    label_text = 'right';
                else
                    label_text = 'other';
                end
                
            else
                % Fine label test
                label_text = sprintf('(%02d) %3.2f', curTracklet.id, yaw_true);
            end
            
            % label_text = sprintf('%3.2f', yaw);
            x = (curTracklet.x1 + curTracklet.x2)/2;
            y = curTracklet.y1;
            text(x, max(y-5,40), label_text, 'color', occlusionColors{occluded+1}, 'BackgroundColor', 'k', ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontWeight', 'bold', 'FontSize', 8);
            
        end
        
        if writeVideoOutput
            % saveas(fig, sprintf('vp_results/seq%02d_%03d_%03d/%03d.jpg', sequenceNum, startImageId, endImageId, imageList(i)));
            print(sprintf('vp_results/seq%02d_%03d_%03d/%03d.png', sequenceNum, startImageId, endImageId, imageList(i)), '-dpng', '-r0');
        end
        
        hold off;
        pause(0.1);
    end
    
    
end
