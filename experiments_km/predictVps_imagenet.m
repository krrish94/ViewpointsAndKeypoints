%% Extract features from Imagenet data and perform error analysis


% Declare global variables
globals;

% Compute pose features by running a forward pass of the network
generatePoseFeatures_imagenet('vggJointVps', 'vggJointVps_iter_70000', 224, params.classInds, 0);