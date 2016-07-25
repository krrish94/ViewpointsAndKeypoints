%% Creates shape test cases, given sequences, frames, and car ids


% List of sequences to be used
sequences = {1};
% First frame to be processed for each sequence specified
startFrames = {0};
% Last frame to be processed for each sequence specified
endFrames = {446};
% List of cars to be processed for each sequence specified
carList = {[1:9, 11:13, 17:19, 23, 28, 33:34, 36:41, 43, 46, 49, 52, 54, 56:57, 59, 63, 67, 69, 71, 75, 81, 83:84, 86, 88, 90:92, 96:97]};


% Get sequence information corresponding to all cars in all sequences,
% where the cars are visible for more than 50 frames
% -- TODO: Implement this functionality -- %


% Sanity checks
if length(sequences) ~= length(startFrames)
    disp('Need equal number of start frames as sequences');
    return;
elseif length(sequences) ~= length(endFrames)
    disp('Need equal number of end frames as sequences');
    return;
elseif length(startFrames) ~= length(endFrames)
    disp('Need equal number of end frames and start frames')
    return;
end

if length(carList) ~= length(sequences)
    disp('Need equal number of car lists and sequences');
end

% For each sequence in the list
for i = 1:length(sequences)
    shapeTestCaseParams.seqNum = sequences{i};
    disp(sprintf('Processing seq: %d', sequences{i}));
    if endFrames{i} - startFrames{i} < 0
        disp('End frame cannot occur after start');
        disp(sprintf('Error in seq %d, start: %d, end: %d', sequences{i}, startFrames{i}, endFrames{i}));
        return;
    end
    
    % Variables for displaying progress
    reverseStr = '';
    numFrames = 0;
        
    curCarList = carList{i};
    % For each frame in the sequence
    for j = startFrames{i}:endFrames{i}
        numFrames = numFrames + 1;
        if mod(numFrames,10) == 0 && startFrames{i} ~= endFrames{i}
            percentDone = numFrames*100/(startFrames{i}-endFrames{i});
            msg = sprintf('Done: %3.1f', percentDone);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end
        
        shapeTestCaseParams.curFrameNum = j;
        % For each car to be processed
        for k = 1:length(curCarList)
            shapeTestCaseParams.curCarId = curCarList(k);
            % Run the script that creates the test case
            createShapeTestCase;
        end
    end
end
