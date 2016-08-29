function [] = mainKpsPredict()

%% Create dataStructs for test

% For each class that we're testing for
for c = params.classInds
    % Display class index
    disp(c);
    % Get the class label fromt the index
    class = pascalIndexClass(c);
    % Read keypoint data for the corresponding class
    readKpsData(class);
end
    
%% Compute predictions for objects with known ground-truth box (Localization setting)

% Extract 'conv6' features using the fine-grained network
extractRigidKeypointFeatures(params.classInds,192,'vggConv6Kps','vggConv6Kps',15);
% Extract 'conv12' features using the coarse network
extractRigidKeypointFeatures(params.classInds,384,'vggConv12Kps','vggConv12Kps',15);
% Extract viewpoint estimates using the viewpoint network (used as a pose
% prior on keypoint locations)
generateKeypointPoseFeatures('vggJointVps','vggJointVps',224,params.classInds,1);


%% Compute predictions for R-CNN detections (Detection setting) 

% We assume pose for detections is precomputed. It not already computed,
% run 'vggJointVps' for viewpoint prediction.

% generateDetectionPoseFeatures(params.classInds,'vggJointVps','vggJointVps',224,1);  
% generateDetectionKpsFeatures(params.classInds,'vggConv6Kps','vggConv6Kps',192,15);
% generateDetectionKpsFeatures(params.classInds,'vggConv12Kps','vggConv12Kps',384,15);
    
end