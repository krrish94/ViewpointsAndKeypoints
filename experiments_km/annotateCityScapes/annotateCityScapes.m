%% Script to run a GUI to annotate samples from the CityScapes dataset


% Path to the directory in which images are contained
dirToBeAnnotated = '/home/km/datasets/cityscapes-train';

% Path where annotations are to be stored
dirToStoreAnnotations = '/home/km/datasets/cityscapes-kps';

% Number of keypoints
numKps = 14;

% Get a directory listing of dirToBeAnnotated
fileList = dir(dirToBeAnnotated);

% Initialize a figure for the GUI
% guiFig = figure;

% Load the first figure into the GUI
% img = imread(fullfile(dirToBeAnnotated, fileList(3).name));
% imshow(img);

% Get the size of the image
imSize = size(img);

% Set figure axis limits
% axis([0, imSize(2), -200, imSize(1)]);

% Set plot title
% title('CityScapes Annotator');

% % Write instruction text
% topText = 'left arrow: previous image; right arrow: next image; Esc: exit';
% text(550, -150, topText, 'FontSize', 10, 'FontWeight', 'bold');
% topText2 = 'Select a car by drawing a rectangle. Annotate kps and double-click to complete.';
% text(450, -50, topText2, 'FontSize', 10, 'FontWeight', 'bold');

% Ignore the first two entries of fileList, since they correspond to . and
% .. repectively.
for i = 3:5
    
    % Initialize a figure for the GUI
    guiFig = figure;
    
    % Read in and display the image
    img = imread(fullfile(dirToBeAnnotated, fileList(i).name));
    imshow(img);
    % Size of the image
    imSize = size(img);
    
    % Initialize object id to -1 (object ids start from 0)
    objectID = -1;
    
    % Name of the mat file in which the current object's annotations are to
    % be saved (remove the last 4 chars from fileList(i) as they correspond
    % to the string '.png').
    saveFileName = fileList(i).name(1:end-4);
    
    % Initialize the struct to store the annotation
    annot.imgFileName = fileList(i).name;
    annot.imSize = imSize;
    
    % Last saved object ID
    lastSavedObjectID = -1;
    
    % As long as the user does not press n
    while 1
        
        % Id of the object in the image
        objectID = objectID + 1;
        
        % Variable to store the bounding box [x, y, w, h]
        bbox = [];
        rect = imrect;
        % Add pushbutton to delete rectangle when done
        uicontrol('Style', 'pushbutton', 'String', 'Done',...
            'Position', [20 20 50 20], ...
            'Callback', 'delete(rect)');
        % Loop to keep the code waiting as long as the rectangle is being drawn
        while isvalid(rect)
            p = wait(rect);
            if ~isempty(p)
                rectangle('Position', p, 'LineWidth', 2, 'EdgeColor', 'r');
                bbox = p;
            end
        end
        
        % Store the bounding box annotation
        annot.bbox = [bbox(1), bbox(2), bbox(1)+bbox(3), bbox(2)+bbox(4)];
        
        % Loop to draw keypoints. If a keypoint is not visible, we click
        % outside the bounding box of the car.
        [x, y] = getpts(guiFig);
        
        % Check if the number of keypoints is greater than the prescribed
        % number. Only in that case, proceed.
        if length(x) < numKps
            fprintf('Img: %d, Obj: %d. Insufficient number of keypoints. Need %d.\n', i, lastSavedObjectID, numKps);
            continue;
        end
        keypoints = [x(1:numKps), y(1:numKps)];
        
        % Reject keypoints outside the bounding box
        for j = 1:numKps
            if keypoints(j,1) < annot.bbox(1) || keypoints(j,1) > annot.bbox(3) || ...
                    keypoints(j,2) < annot.bbox(2) || keypoints(j,2) > annot.bbox(4)
                keypoints(j,:) = [nan, nan];
            end
        end
        
        % Store the final keypoints in the annotation struct
        annot.keypoints = keypoints;
        
        try
            waitforbuttonpress;
        catch
            fprintf('Window closed. Exitting ...\n');
            break;
        end
        key = get(gcf, 'CurrentCharacter');
        switch lower(key)
            case 'q', break;
            case '-', i = max(i-2,0); break;
            case 'n', break;
            % Save the current annotation
            case 's'
                lastSavedObjectID = lastSavedObjectID + 1;
                annot.objID = lastSavedObjectID;
                save(fullfile(dirToStoreAnnotations, [saveFileName, '_', num2str(annot.objID)]), 'annot');
                continue;
            case 'd'
                lastSavedObjectID = lastSavedObjectID + 1;
                annot.objID = lastSavedObjectID;
                save(fullfile(dirToStoreAnnotations, [saveFileName, '_', num2str(annot.objID)]), 'annot');
                break;
        end
    end
end
