function [] = pascalKpsMulticlassTrainValCreate()
% PASCALKPSMULTICLASSTRAINVALCREATE  Generates the text file that is output
% to the data layer of the keypoint network.


% Declare global variables
globals;


% We're interested only in the 'car' class, as of now.
classInds = params.classInds;
% Index at which keypoints for the specific class begin
kpIndexStart = zeros(20,1);
% Number of keypoints for each class
kpNums = zeros(20,1);

% Initialize keypoint metadata for each class
totalKps = 0;
% For each class that we're interested in
for c = classInds
    % Get partnames for the current class
    var = load(fullfile(cachedir,'partNames', pascalIndexClass(c)));
    % Number of keypoints in the current class
    kpNums(c) = length(var.partNames);
    % Index at which the keypoints of the current class start
    kpIndexStart(c) = totalKps;
    % Total number of keypoints
    totalKps = totalKps + length(var.partNames);
end

% Whether or not to flip certain keypoints
flipKps = zeros(1,totalKps);
% Total number of keypoints to flip
totalCt = 0;
% For each class that we're interested in
for c = classInds
    % Get the partnames for the current class
    var = load(fullfile(cachedir,'partNames', pascalIndexClass(c)));
    % Flip keypoints for the current class (making sure that the right/left
    % labels are now reversed appropriately)
    flipKps(totalCt+[1:length(var.partNames)]) = totalCt + findKpsPerm(var.partNames);
    % Total number of flipped keypoints
    totalCt = totalCt + length(var.partNames);
end

% Load the train/val split for pascal
load(fullfile(cachedir,'pascalTrainValIds.mat'));

% Create two sets of filenames, for training and validation
fnamesSets = {};
fnamesSets{1} = trainIds;
fnamesSets{2} = valIds;

% Dimensions of the heatmap to be used for multi-scale response fusion
dims = params.heatMapDims;
% Threshold on the heatmap (???)
probThresh = params.heatMapThresh;


%% Generating window files


% Generate only training and validation data
sets = {'Train','Val'};

% For each set (train/val)
for s=1:length(sets)
    % Get the current set
    set = sets{s};
    % If occluded samples are not to be excluded, ensure that another
    % element 'Occluded' is added to the name of the generated text file.
    if(~params.excludeOccluded)
        set = [set 'Occluded'];
    end
    % Display a status message.
    disp(['Generating data for ' set]);
    % Create a text file
    txtFile = fullfile(finetuneKpsDir, [set num2str(dims(1)) '.txt']);
    % Open the text file for writing
    fid = fopen(txtFile,'w+');
    % Get the list of filenames for files in the current set
    fnames = fnamesSets{s};
    % Sequence number of the current file being processed
    count = 0;
    % For each file in the current set
    for j=1:length(fnames)
        % Get the file name
        id = fnames{j};
        % If the corresponding mat file does not exist, continue
        if(~exist(fullfile(rcnnKpsPascalDataDir,[id '.mat']),'file'))
            continue;
        end
        % Load the mat file
        cands = load(fullfile(rcnnKpsPascalDataDir,id));
        % Get good indices (indices whose values can be written out)
        goodInds = ismember(cands.boxClass,classInds);
        % Exclude samples that would not prove useful for training
        if(params.excludeOccluded)
            goodInds = goodInds & ~cands.occluded & ~cands.truncated & ~cands.difficult ;
        else
             goodInds = goodInds & ~cands.difficult;
        end
        
        % Class label
        cands.boxClass = cands.boxClass(goodInds);
        % Keypoint annotations
        cands.kps = cands.kps(goodInds);
        % Bounding boxes
        cands.bbox = cands.bbox(goodInds,:);
        % Overlap ratio
        cands.overlap = cands.overlap(goodInds,:);
        
        % Number of possible candidates
        numposcands = size(cands.bbox,1);
        numcands = numposcands;
        % If no candidates exist, continue
        if(numcands == 0)
            continue;
        end
        
        % If candidates exist, begin writing them to file
        
        % Increment the sequence number
        count=count+1;
        % Absolute path to the image file
        imgFile = fullfile(pascalImagesDir, [id '.jpg']);
        % Dimensions of the image
        imsize = cands.imsize;
        % Write out image metadata to the file
        % sequence number, abs path to img file, num channels, num rows,
        % numcols, totalKps*heatmapRows*heatmapCols, number of candidates
        fprintf(fid,'# %d\n%s\n%d\n%d\n%d\n%d\n%d\n',count-1,imgFile,3,imsize(1),imsize(2),totalKps*dims(1)*dims(2),numcands);
        % For each possible candidate
        for n = 1:numposcands
            % Normalize keypoint coordinates
            [kpNum, kpCoords] = normalizeKps(cands.kps{n},cands.bbox(n,:),dims);
            [kpNum,kpCoords,kpVal] = gaussianKps(kpNum,kpCoords,dims,probThresh);
            kpNum = kpNum + kpIndexStart(cands.boxClass(n));
            fprintf(fid,'%d %.3f %d %d %d %d %d %d %d',cands.boxClass(n),cands.overlap(n),cands.bbox(n,1),cands.bbox(n,2),cands.bbox(n,3),cands.bbox(n,4),numel(kpNum),kpIndexStart(cands.boxClass(n))*dims(1)*dims(2),kpIndexStart(cands.boxClass(n))*dims(1)*dims(2)+kpNums(cands.boxClass(n))*dims(1)*dims(2)-1);
            for k=1:numel(kpNum)
                %print array index as 1d location
                kpInd = (kpNum(k)-1)*dims(1)*dims(2) + (kpCoords(k,2)-1)*dims(1) + (kpCoords(k,1)-1);
                flipKpInd = (flipKps(kpNum(k))-1)*dims(1)*dims(2)+ (kpCoords(k,2)-1)*dims(1)+ (dims(1) - kpCoords(k,1));
                fprintf(fid,' %d %d %.2f',kpInd, flipKpInd, kpVal(k));
                %print array index as [kpNum,x,y]
                %fprintf(fid,' %d %d %d',kpNum(k),kpCoords(k,1),kpCoords(k,2));
            end
            %if(numel(kpNum)==0)
            %    disp('oops');
            %end
            fprintf(fid,'\n');           
         end
    end
    disp(count);
end

end