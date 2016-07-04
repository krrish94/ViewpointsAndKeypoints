function filteredYaw = testMedianFilter(predictedYaw, windowSize, stridelength)
% TESTMEDIANFILTER  Performs median filtering of predicted azimuths over a 
% sequence


%% Set Filter parameters

if nargin == 1
    % Window size for the median filter
    windowSize = 5;
    % Stride length for the median filter
    strideLength = 1;
    % Currently, padding is not supported
    padding = 0;
elseif nargin == 2
    strideLength = 1;
end


%% Apply the filter

% Initialize the output array
filteredYaw = predictedYaw;

% Determine the start index of the median filter
if length(predictedYaw) < windowSize
    disp('Cannot perform median filtering on such a small array, using the specified filter window size');
    return;
end

% Apply the median filter (assumes that the window size is an odd number)
startIdx = round((windowSize-1)/2) + 1;
endIdx = length(predictedYaw) - round((windowSize-1)/2);

for i = startIdx:endIdx
    % Replace the value at index i by the median of values in the window
    sortedWindow = sort(predictedYaw(i-round((windowSize-1)/2):i+round((windowSize-1)/2)));
    filteredYaw(i) = sortedWindow(round((windowSize-1)/2 + 1));
    % If you want to perform filtering with replacement, uncomment the line
    % below.
    % predictedYaw(i) = sortedWindow(round((windowSize-1)/2 + 1));
end



end
