function [] = generatePascalImageAnnotations()
%GENERATEIMAGEANNOTATIONS Summary of this function goes here
%   Detailed explanation goes here

% Declaring global variables
globals;

% Add to path the directory containing helper functions released by PASCAL
% VOC people
addpath(fullfile(pascalDir,'VOCcode'));
% Class indices for which annotations have to be generated
% classIds = 1:20;
% Since, we're interested only in cars
classIds = 7;
% Delete any '.mat' files in the annotation directory
delete(fullfile(annotationDir,'*.mat'))

% Get names of each annotation xml file in the PASCAL annotation directory
fnames = getFileNamesFromDirectory(fullfile(pascalDir,'VOC2012','Annotations'),'types',{'.xml'},'mode','path');

% For each file
for i=1:length(fnames)
    % Display progress, after every 100 files
    if(~mod(i,100))
        disp([num2str(i) '/' num2str(length(fnames))]);
    end
    % Chop of the last 4 characters, i.e., the '.xml' suffix and retain the
    % remaining as it denotes the id of the image
    id = fnames{i}(end-14:end-4);
    % Get the contents of the XML file to a Matlab struct
    rec = VOCreadxml(fnames{i});
    % Number of bboxes in the current image
    N = length(rec.annotation.object);
    % Cells to store vertices of the rects
    poly_x = {};
    poly_y = {};
    % Vector to store the ID of each rectangle
    voc_rec_id = zeros(N,1);
    % Cell to hold the keypoints
    kps = {};
    % Vector to hold class labels
    class = zeros(N,1);
    % Matrix to store bboxes
    bbox = zeros(N,4);
    % Vectors to hold other status (occlusion, truncation, etc.)
    occluded = zeros(N,1);
    difficult = zeros(N,1);
    truncated = zeros(N,1);
    % Get the size of the detection
    imsize = [str2num(rec.annotation.size.height) str2num(rec.annotation.size.width)];
    % For each detection
    for j =1:length(rec.annotation.object)
        object = rec.annotation.object(j);
        poly_x{j} = [];poly_y{j} = [];kps{j} = [];
        voc_rec_id(j) = j;
        class(j) = pascalClassIndex(object.name);
        if(isfield(object,'occluded'))
            occluded(j) = str2num(object.occluded);
        end
        if(isfield(object,'difficult'))
            difficult(j) = str2num(object.difficult);
        end
        if(isfield(object,'truncated'))
            truncated(j) = str2num(object.truncated);
        end
        if(isfield(object,'bndbox'))
           bbox(j,:) = round([str2num(object.bndbox.xmin) str2num(object.bndbox.ymin) str2num(object.bndbox.xmax) str2num(object.bndbox.ymax)]);
        end
    end
    save(fullfile(annotationDir,[id '.mat']),'poly_x','poly_y','voc_rec_id','kps','class','bbox','difficult','truncated','occluded','imsize');
end

%% Adding kp annotations

% For each class
for c = classIds
    % Get the label of the class and display it
    objClass = pascalIndexClass(c);
    disp(objClass);
    % Load annotations for the corresponding class
    load(fullfile(segkpAnnotationDir,objClass));
    % For each image of the class
    for i=1:length(keypoints.voc_image_id)
        % Load the image
        imName = keypoints.voc_image_id{i};
        imFile = fullfile(annotationDir,[imName '.mat']);
        var = load(imFile);
        % Find rec_id in the image
        index = find(ismember(var.voc_rec_id,keypoints.voc_rec_id(i)));
        if(isempty(index))
            disp('Error : rec_id not found !');
        else
            ind = index;
        end
        if(c ~= var.class(ind))
            disp('oops. Stop now !! ')
        end
        var.kps{ind} = squeeze(keypoints.coords(i,:,:));
        %bbox(ind,:) = keypoints.bbox(i,:); % bbox computed using pascalAnnotations
        %save(imFile,'poly_x','poly_y','voc_rec_id','kps','class','bbox','difficult','truncated','occluded','imsize');
        save(imFile,'-struct','var');
    end
end

%% Adding segm annotations
% for c = classIds
%     objClass = pascalIndexClass(c);
%     disp(objClass);
%     load(fullfile(segkpAnnotationDir,objClass));
%     for i=1:length(segmentations.voc_image_id)
%         imName = segmentations.voc_image_id{i};
%         imFile = fullfile(annotationDir,[imName '.mat']);
%         load(imFile);
%         index = find(ismember(voc_rec_id,segmentations.voc_rec_id(i)));
%         if(isempty(index))
%             %ind = size(bbox,1)+1;
%             %kps{ind} = [];
%             %bbox(ind,:) = zeros(1,4);
%             %voc_rec_id(ind) = segmentations.voc_rec_id(i);
%             %class(ind) = c;
%             disp('Error : rec_id not found !');
%         else
%             ind = index;
%         end
%         poly_x{ind} = segmentations.poly_x{i};
%         poly_y{ind} = segmentations.poly_y{i};
%         save(imFile,'poly_x','poly_y','voc_rec_id','kps','class','bbox','difficult','truncated','occluded','imsize');
%     end
% end

end

