function [] = mainVpsPreprocess()
% MAINVPSPREPROCESS  Creates a text file to carry out training of the
% viewpoint CNN (using coarse and fine bins as mentioned in the paper)


% Declares a set of global variables
globals;

% Returns a string containing the class label (here, 'car')
class = pascalIndexClass(params.classInds);
% Read in data for the car class
disp('Reading data');

% Read viewpoint data corresponding to the current class
readVpsData(class);

% Create a data structure for each image that generates the CNN window file
% used for training. Data is collected from both Imagenet and Pascal.
collectViewpointData();

% Create CNN training file(s)

% Generates window file for network that estimates all three euler angles
createVNetTrainingFile('allData');
% Generates window file for network that estimates azimuth in various bins
% as desired by PASCAL 3D+ evaluation
rcnnMultibinnedJointTrainValTestCreate([24 16 8 4]);

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

