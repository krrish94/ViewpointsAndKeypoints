function [] = getSingleViewDepthStats(expectedDepth, predictedDepth, infoStruct, saveDepthPlots)
% GETSINGLEVIEWDEPTHSTATS  Computes single-view depth statistics,
% given the expected (true) depths, predicted depths, and a general
% information containing sequence number, car id, and the indices of the
% first and last frames processed.


% Declaring global variables
globals;

% Whether or not to save the generated plots
if nargin == 3
    saveDepthPlots = false;
end

% If they have to be saved, specify the directory name
if saveDepthPlots
    mkdirOptional(fullfile(resultsDir, 'vp_results', 'singleViewDepth'));
    singleViewDepthResultsdir = fullfile(resultsDir, 'vp_results', 'singleViewDepth');
end

% Make a consistency plot (plot expected yaw and predicted yaw)
figureHandle = figure;
plot(expectedDepth, 'g', 'LineWidth', 2);
hold on;
plot(predictedDepth, 'r', 'LineWidth', 2);
hold off;
legend({'True', 'Predicted'});
xlabel(sprintf('Time (Frame number)'));
ylabel('Depth (Z-coordinate) in meters');
title(sprintf('Consistency of Depth prediction over frames \n Car ID %02d, seq %02d, frames: %03d-%03d', infoStruct.carId, infoStruct.seqNum, infoStruct.firstFrameId, infoStruct.lastFrameId));
% text(50,22, sprintf('Number of accurate predictions: %03d/%03d', sum(expectedYaw == predictedYaw), length(expectedYaw)));

if saveDepthPlots
    saveas(figureHandle, sprintf('%s/seq%2d_car%2d_noFilter.jpg', singleViewDepthResultsdir, infoStruct.seqNum, infoStruct.carId));
end

% % Different median filter window sizes to try out
% windowSizes = [5, 7, 15, 21];
% for i = 1:length(windowSizes)
%     filteredYaw = testMedianFilter(predictedDepth, windowSizes(i));
%     figureHandle = figure;
%     plot(expectedDepth, 'g', 'LineWidth', 2);
%     hold on;
%     plot(filteredYaw, 'r', 'LineWidth', 2);
%     hold off;
%     legend({'True', sprintf('Filtered (windowsize: %d)', windowSizes(i))});
%     xlabel(sprintf('Time (Frame number)'));
%     ylabel('Depth (Z-coordinate) in meters');
%     title(sprintf('Consistency of Azimuth prediction over frames \n Car ID %02d, seq %02d, frames: %03d-%03d \n Number of accurate predictions: %03d/%03d', infoStruct.carId, infoStruct.seqNum, infoStruct.firstFrameId, infoStruct.lastFrameId, sum(expectedDepth == filteredYaw), length(expectedDepth)));
%     
%     if saveDepthPlots
%         saveas(figureHandle, sprintf('%s/seq%2d_car%2d_medianFilter_windowSize%02d.jpg', singleViewDepthResultsdir, infoStruct.seqNum, infoStruct.carId, windowSizes(i)));
%     end
%     
% end

% Close all windows
close all;

end