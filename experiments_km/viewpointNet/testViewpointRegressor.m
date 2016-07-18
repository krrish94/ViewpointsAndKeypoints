%% Evaluates the viewpoint regressor network on PASCAL VOC


% Declare global variables
globals;


%% Initialize network parameters (MatCaffe)

% Whether or not to (re-)initialize the network
shouldInitialize = false;

if shouldInitialize

    % Name of the subfolder in the 'prototxts' directory where the deployment
    % version of the network is defined, i.e., deploy.prototxt is present
    proto = 'vggViewpointRegressor';
    % Path to prototxt file of the deployment version of the net
    protoFile = fullfile(prototxtDir, proto, 'deploy.prototxt');
    % Name of the caffemodel file (excluding the '.caffemodel' suffix)
    weightsFile = 'net_sgd_iter_65000';
    % Path to the snapshot of the net weights (caffemodel file)
    binFile = fullfile(basedir, 'snapshots', proto, [weightsFile '.caffemodel']);
    
    % Create an RCNN model
    cnn_model = rcnn_create_model(protoFile, binFile);
    % Initialize the model with the weights
    cnn_model = rcnn_load_model(cnn_model);
    % Default image input size is 224 x 224 x 3 (H x W x C)
    inputSize = 224;
    % Initialize the input size of the CNN
    cnn_model.cnn.input_size = inputSize;
    padRatio = 0.00;
    
    % Mean Image (to be subtracted from the test image, for normalization)
    meanNums = [102.9801,115.9465,122.7717]; %magical numbers given by Ross
    meanIm = zeros(inputSize, inputSize, 3);
    for i=1:3
        meanIm(:,:,i) = ones(inputSize)*meanNums(i);
    end
    cnn_model.cnn.image_mean = single(meanIm);
    cnn_model.cnn.batch_size=20;

end


%% Get Pascal validation ids

% Loading the train/test features and labels for the 'car' class
data = load(fullfile(cachedir, 'evalSets', 'car'));

% Encode test labels (to Euler angles)
testLabels = encodePose(data.test.eulers, 'euler');

% Number of test instances
numTest = length(data.test.voc_ids);

% For each test instance
for i = 1:100
    
    % Create a data struct required by the regressor
    curDataStruct.fileName = fullfile(pascalImagesDir, [data.test.voc_ids{i}, '.jpg']);
    curDataStruct.bbox = single(data.test.bboxes(i,:));
    curDataStruct.labels = single(pascalClassIndex('car'));
    
    % Run the network and obtain the predictions
    preds = runNetOnce(cnn_model, curDataStruct);
    
    % Get the azimuth from the predictions
    pred_azimuth = atan2(preds(1), preds(2));
    pred_azimuth = pred_azimuth*180/pi;
    
    % Get the ground-truth azimuth
    true_eulers = data.test.eulers(i,:);
    true_azimuth = true_eulers(3)*180/pi
    
    % Compute error
    pred_error = abs(pred_azimuth - true_azimuth);
    if pred_error >= 360
        pred_error = pred_error - 360;
    end
    
end
