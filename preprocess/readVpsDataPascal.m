function pos = readVpsDataPascal(cls, excludeOccluded)
% READVPSDATAPASCAL  Read pascal to generate filtered instances of classes
% Also reads PASCAL3D dataset to figure out the object index in pascal3D


% Declaring global variables
globals;

% Get the directory containing Pascal 3D annotations for the current class
annoDir = fullfile(PASCAL3Ddir,'Annotations',[cls '_pascal']);
% Return all files of type 'mat' from the annotation directory
posNames = getFileNamesFromDirectory(annoDir,'types',{'.mat'});
% Truncate the last 4 characters of each file name (which correspond to
% '.mat'). Commented out the 'for loop' version, and wrote a vectorized
% form using 'cellfun'.
% % 'Scalar version', slow
% for i=1:length(posNames)
%     posNames{i} = posNames{i}(1:end-4);
% end
% Vectorized version, fast
posNames = cellfun(@(in) in(1:end-4), posNames, 'UniformOutput', false);

% Array to hold values to be returned
pos = [];

% Number of objects stored in the 'pos' array
numpos = 0;
% For each mat file stored in posNames
for j = 1:length(posNames)
    % Load the mat file
    load(fullfile(annoDir,posNames{j}));
    % For each bbox detection in the mat file
    for k=1:length(record.objects)
        % If the object belongs to the current class, and has its viewpoint
        % annotated, and is not one of the instances classified 'difficult'
        % according to Pascal 3D evaluation
        if(strcmp(cls,record.objects(k).class) && ~isempty(record.objects(k).viewpoint) && ~record.objects(k).difficult)
            % If we want to include occluded objects, all objects are
            % stored in the 'pos' array. Else, only non-occluded and
            % non-truncated objects are stored.
            if(~ excludeOccluded || (~record.objects(k).truncated && ~record.objects(k).occluded))
                % Increment the count of the number of objects stored
                numpos = numpos + 1;
                % Bounding box for the instance
                bbox   = round(record.objects(k).bbox);
                % Size of the current image
                pos(numpos).imsize = record.imgsize(1:2);
                % voc_id
                pos(numpos).voc_image_id = posNames{j};
                % voc_rec_id
                pos(numpos).voc_rec_id = k;
                %pos(numpos).im      = [VOCopts.datadir rec.imgname];
                % bbox
                pos(numpos).bbox   = bbox;
                % view (unused in many cases)
                pos(numpos).view    = '';
                % Array to store keypoints
                pos(numpos).kps     = [];
                % Cell to store part names
                pos(numpos).part_names  = {};
                %pos(numpos).maskBbox        = keypoints.bbox(ki,:);
                % Segmentation masks (in the form of a polygon)
                pos(numpos).poly_x      = [];
                pos(numpos).poly_y      = [];
                pos(numpos).mask = [];
                % Class label
                pos(numpos).class       = cls;
                % Rotation matrix that expresses the body-centric (global)
                % frame in the camera coordinate frame
                pos(numpos).rot = [];
                % Euler angles (azimuth, elevation, cyclorotation)
                pos(numpos).euler = [];
                % Detection score
                pos(numpos).detScore = Inf;
                % IoU score
                pos(numpos).IoU = 1;
                % Occlusion/truncation status
                pos(numpos).occluded = record.objects(k).occluded;
                pos(numpos).truncated = record.objects(k).truncated;

                % Index of the instance in the current image
                objectInd = k;
                pos(numpos).objectInd = objectInd;
                pos(numpos).dataset = 'pascal';
                % Get viewpoint data (including camera intrinsics)
                viewpoint = record.objects(objectInd).viewpoint;
                % Get rotation matrix and Euler angles from the viewpoint
                % struct
                [rot, euler]=viewpointToRots(viewpoint);
                % Store them in the 'pos' array
                pos(numpos).rot=rot;
                pos(numpos).euler=euler;
                % Subtype of the current object, i.e., index of the 'most'
                % visually similar CAD model to the current instance
                pos(numpos).subtype = record.objects(k).cad_index;
            end
        end
    end
end

end


% Function to convert viewpoint predictions/annotations from spherical
% coordinates to a rotation matrix
function [R, euler] = viewpointToRots(vp)
    
    if(~isfield(vp,'azimuth'))
        vp.azimuth = vp.azimuth_coarse;
    end
    
    if(~isfield(vp,'elevation'))
        vp.elevation = vp.elevation_coarse;
    end
    
    if(~isfield(vp,'theta'))
        vp.theta = 0;
    end
    euler = [vp.azimuth vp.elevation vp.theta]' .* pi/180;
    R = angle2dcm(euler(3), euler(2)-pi/2, -euler(1),'ZXZ'); %took a lot of work to figure this formula out !!
    euler = euler([3 2 1]);
end
