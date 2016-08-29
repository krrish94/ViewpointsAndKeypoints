function [] = mainKpsPreprocess()
% MAINKPSPREPROCESS  Performs all preprocessing required to train the
% keypoint prediction network.


% Declaring global variables
globals;

% Generate annotations over Pascal3D. Store them in annotationDir.
generatePascalImageAnnotations();

% Generate RCNN data files (currently only for the 'car' class)
rcnnKpsDataCollect();

% Generate 'partName' labels
mkdirOptional(fullfile(cachedir,'partNames'));
for c = params.classInds
    class = pascalIndexClass(c);
    var = load(fullfile(segkpAnnotationDir,class));
    partNames = var.keypoints.labels;
    save(fullfile(cachedir,'partNames',class),'partNames');
end

% Generate window file for Conv6Kps
params.heatMapDims = [6 6];
pascalKpsMulticlassTrainValCreate()
% Generate window file for Conv12Kps
params.heatMapDims = [12 12];
pascalKpsMulticlassTrainValCreate()

% Train the cnns (not in MATLAB, unfortunately!)

end