%% Script to predict azimuth on Thomas Brox's Car dataset (ICCV 2015)


% Base directory of the Brox dataset
broxDataDir = fullfile(basedir, 'data', 'Brox');
% Directory containing annotations
broxAnnotationDir = fullfile(broxDataDir, 'annotations');

% ID(s) of the car(s) we want to test on
carId = [2];

% Initialize the viewpoint network
% initViewpointNet;

% pascalBins = -171.43 + 17.14*[0:20];
% pascalBins(find(pascalBins < 0)) = pascalBins(find(pascalBins < 0)) + 360

% For each car that we want to predict azimuth for
for i = 1:length(carId)
    
    % Load the annotation file for the current car
    broxAnnotationFile = fopen(sprintf('%s/%d_annot.txt', broxAnnotationDir, carId(i)), 'r');
    % Read annotations from file. Each column is stored in a single cell
    % array.
    broxAnnotations = textscan(broxAnnotationFile, '%s %d %d %d %d %d');
    
    % For each image of the current car
    for j = 1:length(broxAnnotations{1})
        % Path to the image file
        imFile = fullfile(broxDataDir, broxAnnotations{1}{j});
        % Read in the image file
        img = imread(imFile);
        % Resize the image by a scale factor of 0.5
        imgDisplay = imresize(img, 0.5);
        % Get the bounding box annotations. Convert to single precision, as
        % the defined Caffe layer expects single-precision values.
        bbox = [broxAnnotations{2}(j), broxAnnotations{3}(j), broxAnnotations{4}(j), broxAnnotations{5}(j)];
        bbox = single(bbox);
        % Get the true azimuth
        yawTrue = broxAnnotations{6}(j);
        
        % Create a data struct to pass to the viewpoint net
        dataStruct.fileName = imFile;
        dataStruct.bbox = bbox;
        % Label for the 'car' class in Pascal is 7
        dataStruct.labels = 7;
        
        % Run the network and obtain the prediction
        featVec = runNetOnce(cnn_model, dataStruct);
        
        % Compute predicted pose from the returned feature vector
        yaw = getPoseFromFeat_test(featVec);
        
        % Display the image
        imshow(imgDisplay);
        [yawTrue, yaw, pascalBins(yaw)]
        pause;
    end
end
