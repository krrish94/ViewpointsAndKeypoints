%% Performs Keypoint Localization (Assumes bboxes are provided)

% clear;
% startup;

% Declaring global varaibles
globals;

% Loading the train/test split for Pascal
load(fullfile(cachedir,'pascalTrainValIds'));

% Matrix to hold performance parameters for each of the 20 classes
% perf = zeros(20,5);
% But, we're here interested only in the 'car' class
perf = zeros(1,5);

% For each class
for c = params.classInds
    % Get the class label and display it
    class = pascalIndexClass(c);
    disp(class)
    % Dimensions of the heatmap
    params.heatMapDims = [24 24];
    % Network for detecting keypoints
    params.kpsNet = 'vgg';
    
    % Loads conv6 and conv12 features
    loadFeatRigid;
    
    % Load pose priors
    [priorFeat] = posePrior(dataStruct,class,trainIds);
    
    %% feat
    featStruct{1} = feat6;
    featStruct{2} = feat12;
    featStruct{3} = feat12+feat6;
    featStruct{4} = (feat12+feat6) + log(priorFeat+eps);
    
    %% pred
    
    dataStruct.pascalbox = dataStruct.bbox; %hack to make pck Evaluation happy
    params.predMethod = 'maxLocation';
    params.alpha = 0.1;
    acc = [];
    for i=1:(length(featStruct))
        feat = featStruct{i};
        pred = predictAll(feat,dataStruct);
        %acc(i) = mean(pckMetric(pred,dataStruct,valIds));
        acc(i) = mean(pckMetric(pred,dataStruct,ismember(dataStruct.voc_image_id,valIds) & ~dataStruct.occluded));
    end
    ningPreds = getNingPreds(class,dataStruct);
    acc(i+1) = mean(pckMetric(ningPreds,dataStruct,ismember(dataStruct.voc_image_id,valIds) & ~dataStruct.occluded));
    disp(acc)
    perf(c,:) = acc;
end
