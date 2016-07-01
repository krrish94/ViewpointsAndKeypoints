function [rotationData] = readVpsData(cls)
% READVPSDATA  Reads rotation data for the current class from Imagenet and
% Pascal 3D


% Declare global variables
globals;

% Read viewpoint data from the PASCAL3D dataset
pascalData = readVpsDataPascal(cls, params.excludeOccluded);
fname = fullfile(rotationPascalDataDir,cls);
rotationData = pascalData;
save(fname,'rotationData');

% Read viewpoint data from the Imagenet dataset
imagenetData = readVpsDataImagenet(cls);
fname = fullfile(rotationImagenetDataDir,cls);
rotationData = imagenetData;
save(fname,'rotationData');

% Concatenate both the data matrices
rotationData = horzcat(pascalData,imagenetData);
fname = fullfile(rotationJointDataDir,cls);
save(fname,'rotationData');

end

