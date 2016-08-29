function [] = createVNetTrainingFile(fNameSuffix)
% CREATEVNETTRAININGFILE  Create a training file for the viewpoint network
%   fNameSuffix is the name to be given to the VNet train file.

% Format of the window file created
%   repeat:
%       absolute image path
%       channels
%       height
%       width
%       number of windows
%       classIdx overlap x1 y1 x2 y2 regionIdx [poseParams 1..N]


% Declaring global variables
globals;

% Generate train filenames
load(fullfile(cachedir, 'imagenetTrainIds.mat'));
load(fullfile(cachedir, 'pascalTrainValIds.mat'));

% Concatenate Pascal train ids, and Imagenet train filenames

% Use the following for training
fileNames = unique(vertcat(trainIds, fnamesTrain));
% Use the following for validation
% fileNames = unique(vertcat(valIds));

disp('Generating CNN window file');

% Open (create) the text file (for writing)
txtFile = fullfile(VNetTrainFilesDir, [fNameSuffix '.txt']);
fid = fopen(txtFile, 'w+');

count = 0;

% Variables to display progress text
reverseStr = '';

% Write data to file
for j = 1:length(fileNames)
    
    % Display progress (after every 100 iters)
    if mod(j,100) == 0
        percentDone = j*100/length(fileNames);
        msg = sprintf('Done: %3.1f', percentDone);
        fprintf([reverseStr, msg]);
        reverseStr = repmat(sprintf('\b'), 1, length(msg));
    end
    
    % Current fileName
    id = fileNames{j};
    if exist(fullfile(viewpointDataDir,[id '.mat']), 'file')
        % File path
        candFile = fullfile(viewpointDataDir, [id '.mat']);
    else
        continue;
    end
    % Load candidates
    cands = load(candFile);
    if isempty(cands.overlap)
        continue;
    end
    
    % Hack by KM
    params.candidateThresh = 1;
    %
    
    numCands = round(sum(cands.overlap >= params.candidateThresh));
    if numCands == 0
        continue;
    end
    count = count + 1;
    imSize = cands.imSize;
    [imgDir, imgExt] = getDatasetImgDir(cands.dataset);
    imgFile = fullfile(imgDir, [id imgExt]);
    
    % Write data in the following format
    %
    % # count-1
    % imgFile (absolute path to image)
    % num channels (usually 3)
    % height
    % width
    % number of candidates (sat n)
    %   n lines, where each line contains info about one candidate
    %   classIndex, overlap, x1, y1, x2, y2, euler1,2,3 (coarse), euler
    %   1,2,3,(fine)
    
    fprintf(fid,'# %d\n%s\n%d\n%d\n%d\n%d\n', count-1, imgFile, ...
            3, imSize(1), imSize(2), numCands);
    
    %if(max(cands.euler(:,1))>=pi/2 || max(cands.euler(:,2)>=pi/2 ))
    %    disp('Oops');
    %end
    for n=1:size(cands.overlap,1)
        
        fprintf(fid, '%d %d %d %d %f %f %f %f\n', cands.bbox(n,1), ...
            cands.bbox(n,2), cands.bbox(n,3), cands.bbox(n,4), ...
            sin(cands.euler(n,3)), cos(cands.euler(n,3)), ...
            sin(cands.euler(n,1)), cos(cands.euler(n,1)));

        
        % Don't use the following file format. Window data layer modified
        % for regression by KM does not understand this format.
        
%         fprintf(fid, '%s %d %d %d %d %d %d %f %f %f %f %f %f', imgFile, ...
%             imSize(1), imSize(2), cands.bbox(n,1), cands.bbox(n,2), ...
%             cands.bbox(n,3), cands.bbox(n,4), sin(cands.euler(n,1)), ...
%             cos(cands.euler(n,1)), sin(cands.euler(n,2)), ...
%             cos(cands.euler(n,2)), sin(cands.euler(n,3)), ...
%             cos(cands.euler(n,3)));

        % Original format (as Shubham defined it)
        
%         fprintf(fid,'%d %f %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n',...
%             cands.classIndex(n),cands.overlap(n),...
%             cands.bbox(n,1),cands.bbox(n,2),cands.bbox(n,3),cands.bbox(n,4),...
%             ceil(cands.euler(n,1)*10.5/pi+9.5),ceil(-cands.euler(n,1)*10.5/pi+9.5),...
%             ceil(cands.euler(n,2)*10.5/pi+9.5),ceil(cands.euler(n,2)*10.5/pi+9.5),...
%             floor(cands.euler(n,3)*10.5/pi),20-floor(cands.euler(n,3)*10.5/pi),...
%             ceil(cands.euler(n,1)*3.5/pi+2.5),ceil(-cands.euler(n,1)*3.5/pi+2.5),...
%             ceil(cands.euler(n,2)*3.5/pi+2.5),ceil(cands.euler(n,2)*3.5/pi+2.5),...
%             floor(cands.euler(n,3)*3.5/pi),6-floor(cands.euler(n,3)*3.5/pi));
    end
end

fprintf('\n');

% Close the file
fclose(fid);

end
