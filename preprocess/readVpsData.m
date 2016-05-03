function [rotationData] = readVpsData(cls)
%READDATA Summary of this function goes here
%   Detailed explanation goes here

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

