function [] = visualizeKpMaps(dataStruct,kps,partNames,sc)
% VISUALIZEKPMAPS  Plots the keypoint heatmaps

% close all;

% Declaring global variables
globals;

% If there are only three inputs, set the plot title to 0
if(nargin<4)
    sc = 0;
end

% Obtain the bounding box and class label information from dataStruct
bbox = dataStruct.bbox;
class = dataStruct.class;

% Get keypoint coordinates relative to the top-left corner of the bbox
x  = kps(1,:) - bbox(1);
y  = kps(2,:) - bbox(2);

% Keypoint feature vector (keypoint heat maps)
maps = dataStruct.kpFeat;
% Get the height, width, and number of keypoints from maps
[H,W,Kp] = size(maps);

% Number of rows in the (sub) plot
plotH = 2;
% Number of cols in the (sub) plot
plotW = ceil((Kp+2)/plotH);

% If neighbor maps exist, initialize them
neighborMaps = zeros(H,W,Kp);
% if(~isfield('neighborFeat',dataStruct))
if(isfield('neighborFeat',dataStruct))
    neighborMaps = dataStruct.neighborFeat;
end


% Reading Image
im = imread(fullfile(dataStruct.fileName));
% Round the bbox to integral vertices
bbox = round(dataStruct.bbox);
% 
deltaX = ceil(max([0,-bbox(1)+1,-size(im,2)+bbox(3)]));
deltaY = ceil(max([0,-bbox(2)+1,-size(im,1)+bbox(4)]));
% Create a blank RGB image
im2 = uint8(zeros(size(im,1)+2*deltaY,size(im,2)+2*deltaX,3));
% Initialize it with the current image being processed
im2(deltaY+[1:size(im,1)],deltaX+[1:size(im,2)],:)=im;
% Crop out only the part corresponding to the detection
im = im2(deltaY+[bbox(2):bbox(4)],deltaX+[bbox(1):bbox(3)],:);
% Convert it to grayscale
img = im2double(im);
img = rgb2gray(img);

% In the first sub-plot, show the RGB image corresponding to the detection
% window
subplot(plotH,plotW,1);
imshow(im);
axis equal;
title(num2str(sc));
hold on;


% %% PASCAL3d
% 
% pascal3Dfile = fullfile(PASCAL3Ddir,'Annotations',[class '_pascal'],[dataStruct.voc_image_id '.mat']); 
% record = load(pascal3Dfile);record = record.record;
% bbox = dataStruct.bbox;
% eulersPred = dataStruct.eulers;
% 
% objectInd = dataStruct.objectInd;
% 
% viewpoint = record.objects(objectInd).viewpoint;
% viewpoint.azimuth = eulersPred(3);
% viewpoint.elevation = eulersPred(2);
% viewpoint.theta = eulersPred(1);
% record.objects(objectInd).viewpoint = viewpoint;
% 
% CADPath = fullfile(PASCAL3Ddir,'CAD',class);
% cad = load(CADPath);
% cad = cad.(class);
% 
% subplot(plotH,plotW,2);
% 
% vertex = cad(record.objects(objectInd).cad_index).vertices;
% face = cad(record.objects(objectInd).cad_index).faces;
% [x2d,Z] = project3d(vertex, record.objects(objectInd),face);
% %     %patch('vertices', [x2d Z], 'faces', face, ...
% %     %    'FaceColor', 'blue', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% axis equal;
% 
% trisurf(face,x2d(:,1),-x2d(:,2),-Z);axis equal; %empirically found this view and plot to work best
% view(0,90);


%% Display map images

% For each keypoint
for kp = 1:Kp
    % Current heatmap
    heatIm = maps(:,:,kp);
    % Current prior map
    neighborIm = imresize(neighborMaps(:,:,kp),[size(im,1),size(im,2)]);
    
    %heatIm = normalizeHeatIm(heatIm,size(im,2)/size(im,1));
    maps(:,:,kp) = heatIm;
    heatIm = imresize(heatIm,[size(im,1),size(im,2)]);

    % Initialize a blank image
    im1=zeros(size(img));
    % Overlay the actual image, heatmap, and prior
    im1(:,:,1)=0.2*img+0.8*heatIm.*neighborIm;
    im1(:,:,2)=0.2*img+0.8*heatIm.*neighborIm;
    im1(:,:,3)=0.2*img;
    im1 = max(im1,0);im1 = min(im1,1);
    subplot(plotH,plotW,kp+2);
    %imagesc(im1);axis equal;
    imshow(im1);axis equal;
    hold on;
    plot(x(kp),y(kp),'r.');hold on;
    xKp = round(x(kp));
    yKp = round(y(kp));
    scKp = 0;

    if(xKp>0 && yKp>0 && xKp<=size(im,2) && yKp<=size(im,1))
        scKp = heatIm(yKp,xKp);
    end

    title([partNames{kp} ' : ' num2str(scKp)]);
    
end

pause();
close all;

end

