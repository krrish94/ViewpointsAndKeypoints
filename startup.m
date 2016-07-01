%% Script to be executed at startup (when Matlab is started from this dir)

% Declaring global variables
globals;

% Root directory of this codebase. Assumes that you're running code from
% this directory.
basedir = pwd();
% Directory where all the intermediate computations and data will be saved
% (cached), hence the name.
cachedir  = fullfile(basedir,'cachedir');


%% These paths need to be set properly


% Root directory containing the Pascal3D dataset
PASCAL3Ddir = fullfile(basedir,'data','PASCAL3D');
% Base directory of Pascal VOC's development kit
pascalDir = '/home/km/code/ViewpointsAndKeypoints/data/PASCAL3D/PASCAL/VOCdevkit/';
% Directory containing Pascal (VOC) images
pascalImagesDir = '/home/km/code/ViewpointsAndKeypoints/data/PASCAL3D/PASCAL/VOCdevkit/VOC2012/JPEGImages';
% Directory containing Imagenet images
imagenetImagesDir = fullfile(basedir,'data','imagenet','images');

% File containing RCNN detections on the validation set for Pascal
rcnnDetectionsFile = fullfile(basedir,'data','VOC2012_val_det.mat');
% Contains keypoint annotations, as well as instance segmentation for
% Pascal images, organized by class. Required for keypoint prediction.
segkpAnnotationDir = fullfile(basedir,'data','segkps');
% Directory where caffemodels are saved - you'll have to set this up
snapshotsDir = fullfile(cachedir,'snapshots');

    
%% The paths below should not be edited

% Important parameters for getting things to work as expected.
params = getParams;

% Directory containing viewpoint annotations generated over Pascal.
% Each annotations includes the image file name, bounding boxes detected,
% class label of each bounding box, and occlusion stats. Organized by image
% file name.
rcnnVpsPascalDataDir = fullfile(cachedir,'rcnnVpsPascalData');
% Directory containing viewpoint annotations for Imagenet, organized by
% image file name.
rcnnVpsImagenetDataDir = fullfile(cachedir,'rcnnVpsImagenetData');
% Directory containing keypoint annotations generated for Pascal, organized
% by image file name.
rcnnKpsPascalDataDir = fullfile(cachedir,'rcnnKpsPascalData');
% Directory containing keypoint annotations generated for Pascal and
% Imagenet, organized by image file name.
viewpointDataDir = fullfile(cachedir, 'viewpointData');

% Directory containing keypoint annotations for Pascal, organized by class
kpsPascalDataDir = fullfile(cachedir,'kpsDataPascal');

% Directory containing pascal image annotations, organized by image name
annotationDir = fullfile(basedir,'data','pascalAnnotations','imgAnnotations');

% Directory containing rotation data for Pascal, organized by class
rotationPascalDataDir = fullfile(cachedir,'rotationDataPascal');
% Directory containing rotation data for Imagenet, organized by class
rotationImagenetDataDir = fullfile(cachedir,'rotationDataImagenet');
% Directory containing joint rotation data (all three Euler angles),
% organized by class
rotationJointDataDir = fullfile(cachedir,'rotationDataJoint');

% Viewpoint training metadata
finetuneVpsDir = fullfile(cachedir,'rcnnFinetuneVps');
VNetTrainFilesDir = fullfile(cachedir, 'VNetTrainFiles');
% Keypoint training metadata
finetuneKpsDir = fullfile(cachedir,'rcnnFinetuneKps');

% Directory where visualizations used for the main paper will be saved
websiteDir = fullfile(cachedir,'visualization');

% Directory containing prototxts of network models
prototxtDir = fullfile(basedir,'prototxts');

% Directories where results will be stored in the subfolders kp_results and
% vp_results
resultsDir = fullfile(cachedir, 'results');


%% Adding certain folders to Matlab path

folders = {'analysisVp','analysisKp','detectionPose','pose','encoding','predict','evaluate','utils','visualization','evaluation','learning','preprocess','rcnnKp','rcnnVp','cnnFeatures', 'experiments_km', 'viewpointNet'};
for i=1:length(folders)
    addpath(genpath(folders{i}));
end

clear i;
clear folders;


% Create cachedir if it does not exist
mkdirOptional(cachedir);
% Load the train/val split for Pascal if it exists
if exist(fullfile(cachedir,'pascalTrainValIds.mat'))
    load(fullfile(cachedir,'pascalTrainValIds'))
% Else, create the Pascal train/val split file using the text files
% train.txt and val.txt, provided by Pascal VOC.
else
    fIdTrain = fopen(fullfile(pascalDir,'VOC2012','ImageSets','Main','train.txt'));
    trainIds = textscan(fIdTrain,'%s');
    trainIds = trainIds{1};
    fIdVal = fopen(fullfile(pascalDir,'VOC2012','ImageSets','Main','val.txt'));
    valIds = textscan(fIdVal,'%s');
    valIds = valIds{1};
    save(fullfile(cachedir,'pascalTrainValIds.mat'),'trainIds','valIds');
end

% Create the imagenet train ids file, if it doesn't already exist.
if ~exist(fullfile(cachedir,'imagenetTrainIds.mat'))
    fnamesTrain = generateImagenetTrainNames();
    save(fullfile(cachedir,'imagenetTrainIds.mat'),'fnamesTrain');
end

% Create required directories if they do not already exist

mkdirOptional(rotationJointDataDir);
mkdirOptional(rotationImagenetDataDir);
mkdirOptional(rotationPascalDataDir);
mkdirOptional(kpsPascalDataDir);
mkdirOptional(annotationDir);

mkdirOptional(rcnnVpsImagenetDataDir);
mkdirOptional(rcnnVpsPascalDataDir);
mkdirOptional(rcnnKpsPascalDataDir);
mkdirOptional(viewpointDataDir);

mkdirOptional(finetuneVpsDir);
mkdirOptional(VNetTrainFilesDir);
mkdirOptional(finetuneKpsDir);

mkdirOptional(websiteDir);

% Check if Caffe has been installed at the appropriate location. Throw a
% warning if the check fails.
if exist('external/caffe/matlab/caffe')
  addpath('external/caffe/matlab/caffe');
else
  warning('Please install Caffe in ./external/caffe');
end
