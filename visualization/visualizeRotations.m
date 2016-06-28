function [] = visualizeRotations(gt,prediction,data,encoding,scale)
%VISUALIZEROTATIONS Summary of this function goes here
%   Detailed explanation goes here


% Declaring global variables
globals;

% Decode the current predictions to Euler angles
eulersPred = decodePose(prediction,encoding);
% Decode the ground-truth predictions to Euler angles
eulersGt = decodePose(gt,encoding);

% Encode the current predictions from Euler angles to rotation matrices
rotsPred = encodePose(eulersPred,'rot');
% Encode the ground-truth predictions from Euler angles to rotation mats
rotsGt = encodePose(eulersGt,'rot');

% For each prediction (in the ground-truch annotations passed)
for i = 1:size(gt,1)
    % Reshape the predicted and ground-truth rotation matrices to 3 x 3
    rotPred = reshape(rotsPred(i,:),3,3);
    rotGt = reshape(rotsGt(i,:),3,3);
    
    % Convert the predicted vector from a rotation matrix to an axis-angle
    % form.
    predVec = vrrotmat2vec(rotPred);
    predVec = predVec(1:3)*predVec(4)*scale;
    % Convert the ground-truth annotation from a rotation matrix to an
    % axis-angle form
    gtVec = vrrotmat2vec(rotGt);
    gtVec = gtVec(1:3)*gtVec(4)*scale;
    
    % Read in the current image from the pascal dataset
    im = imread(fullfile(pascalImagesDir,[data.voc_ids{i} '.jpg']));
    % Retrive the current instance's bounding box
    bbox = data.bboxes(i,:);
    % Get the center of the bounding box
    xMean = (bbox(1)+bbox(3))/2;
    yMean = (bbox(2)+bbox(4))/2;
    
    % Initialize a new plot
    figure();
    
    % Plot the image onto the axes
    warp(im);hold on;
    % 
    quiver3(xMean,yMean,0,predVec(1),predVec(2),predVec(3),'g','LineWidth',3);hold on;
    quiver3(xMean,yMean,0,gtVec(1),gtVec(2),gtVec(3),'r','LineWidth',3);hold on;
    
    % 
    e1 = norm(logm(rotGt*rotPred'))*sqrt(2)*180/pi;
    e2 =  1/scale*norm(gtVec-predVec)*180/pi;
    
    title([num2str(e1) , ' ',num2str(e2), ' ' num2str(e1/e2)]);axis equal;
    pause(); close all;
end

end

