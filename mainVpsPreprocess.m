function [] = mainVpsPreprocess()


% Declares a set of global variables
globals;

% For each of the 12 classes in PASCAL3D, read data
for c = params.classInds
    % Returns a string containg the class label (eg. 'aeroplane', 'car',
    % etc.)
    class = pascalIndexClass(c);
    disp(['Reading data for : ' class]);
    % Read in data for the corresponding class
    readVpsData(class);
end

% Create imagewise datastructures for cnn window file generation
vpsPascalDataCollect();
vpsImagenetDataCollect();

% Create cnn training file(s)
rcnnBinnedJointTrainValTestCreate(''); %generates window file for network that estimates all three euler angles
rcnnMultibinnedJointTrainValTestCreate([24 16 8 4]); % generates window file for network that estimates azimuth in various bins as desired by pascal3D+ evaluation

%% Train the CNNs !
% (not from matlab, unfortunately)
% update the window file paths in the data layers of 
% PATH_TO_PROTOTXT_DIR/vggJointVps/trainTest.prototxt and PATH_TO_PROTOTXT_DIR/vggAzimuthVps/trainTest.prototxt
% to refer to the Train/Val files created by above functions.

% we train two networks here - one for predicting all the euler angles, other for various bin sizes of azimuth as required by AVP evaluation

% the shell scripts look as follows
% ./build/tools/caffe.bin train -solver PATH_TO_PROTOTXT_DIR/vggJointVps/solver.prototxt -weights PATH_TO_PRETRAINED_VGG_CAFFEMODEL
% ./build/tools/caffe.bin train -solver PATH_TO_PROTOTXT_DIR/vggAzimuthVps/solver.prototxt -weights PATH_TO_PRETRAINED_VGG_CAFFEMODEL
% after training the models, save the final snapshot in SNAPSHOT_DIR/finalSnapshots/[vggJointVps,vggAzimuthVps].caffemodel/

    
end

