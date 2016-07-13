function params = getParams()
%GET_PARAMS  Initializes important parameters required for execution

% We're interested only in the 'car' class here. IMPORTANT: Do not define
% it as a variable, it must be a matrix or some functions may not work. Do
% not use 'params.classInds = 7', even if it may seem harmless.
params.classInds = 7;
% The 'car' class has 14 keypoints. Hard-coding it here, as I expect to run
% this code only on cars
params.numKps = 14;

params.vpsDataset = '';
params.features = 'vggJointVpsMirror'; %see cache/features folder to see what features are allowed
params.angleEncoding = 'euler'; %'euler' or 'rot' or 'cos_sin' or 'axisAngle'. 'cos_sin' is highly recommended !
params.optMethod = 'bin'; %'bin' or 'svr' or 'lsq' // don't use 'rf' for now as it breaks down
params.candidateThresh = 0.5; %IoU threshold for candidates to be used to in training. Dont set less than 0.5 as it might assigne same candidate to multiple gts
params.trainValSets = {''}; %Empty String implies Gt
params.testSets = {''}; %CandidatesPool
params.nHypotheses = 1;

params.interpolationMethod = 'cubic'; %cubic works best even though its slightly slow
params.heatMapDims = [24 24]; % [xdim ydim]
params.kpsSuffix = '';
params.rigidKpsDataset = 'p3d'; %'p3d' or 'pascal'
params.kpsNet = 'vggAll';
%params.classInds = [1 2 4 5 6 7 9 14 18 19 20]; %rigid categories
% params.classInds = [1 2 4 5 6 7 9 11 14 18 19 20]; %rigid categories
params.torsoKps = [4 7 1 10]; %for pascal person

params.excludeOccluded = false;

params.visMethod = 'maxLocation'; %  'maxCandidate'/'maxLocation'
params.predMethod = 'maxLocation'; %  'maxCandidate'/'maxLocation'

params.heatMapThresh = 0.2; %used for window file generation
params.alpha = 0.1;
params.numKpCands = 1;
params.rigidApkEvalAlpha = 0.1;

end
