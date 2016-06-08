%% Performs Keypoint Detection (Bounding Boxes detected by RCNN)

%% Initialization

startup;
% Declaring global variables
globals;
% Loading the train test split
load(fullfile(cachedir,'pascalTrainValIds'));
% Heatmap dimensions (usually 12 x 12, for the coarse predictor)
params.heatMapDims = [12 12];

%% Pose Priors

% For each class
for c = params.classInds
    % Get the class label
    class = pascalIndexClass(c);
    % Generate pose prior maps
    posePriorMaps(class,trainIds);
end

% Whether or not to use the pose priors
priorAlphas = [0 0.2];
aps = zeros(numel(priorAlphas,20));

%% Compute APK
for c = params.classInds
    class = pascalIndexClass(c);
    annot = getKeypointannotationStruct(class,valIds);
    preds = computePredictionStruct(class,(unique(annot.img_name)),priorAlphas,'All');
    for d = 1:numel(priorAlphas)
        aps(d,c) = compute_kp_APK(annot,preds{d},0.2);
    end
    disp(aps(:,c))
end

%% Saving
save(fullfile(cachedir,'rigidApkResults'),'aps','priorAlphas');