function [] = getViewpointConsistencyStats(expectedYaw, predictedYaw, infoStruct, saveConsistencyPlots)
% GETVIEWPOITNCONSISTENCYSTATS  Computes viewpoint consistency statistics,
% given the expected (true) azimuths, predicted azimuths, and a general
% information containing sequence number, car id, and the indices of the
% first and last frames processed.


% Declaring global variables
globals;

% Whether or not to save the generated plots
if nargin == 3
    saveConsistencyPlots = false;
end

% If they have to be saved, specify the directory name
if saveConsistencyPlots
    vpConsistencyResultsdir = fullfile(resultsDir, 'vp_results', 'viewpointConsistency');
end

% Make a consistency plot (plot expected yaw and predicted yaw)
figureHandle = figure;
plot(expectedYaw, 'g', 'LineWidth', 2);
hold on;
plot(predictedYaw, 'r', 'LineWidth', 2);
hold off;
legend({'True', 'Predicted'});
xlabel(sprintf('Frame number/Time'));
ylabel('Azimuth bin number (total 21 bins)');
ylim([0 25]);
title(sprintf('Consistency of Azimuth prediction over frames \n Car ID %02d, seq %02d, frames: %03d-%03d \n Number of accurate predictions: %03d/%03d', infoStruct.carId, infoStruct.seqNum, infoStruct.firstFrameId, infoStruct.lastFrameId, sum(expectedYaw == predictedYaw), length(expectedYaw)));
% text(50,22, sprintf('Number of accurate predictions: %03d/%03d', sum(expectedYaw == predictedYaw), length(expectedYaw)));

if saveConsistencyPlots
    saveas(figureHandle, sprintf('%s/seq%2d_car%2d_noFilter.jpg', vpConsistencyResultsdir, infoStruct.seqNum, infoStruct.carId));
end

% Different median filter window sizes to try out
windowSizes = [5, 7, 15, 21];
for i = 1:length(windowSizes)
    filteredYaw = testMedianFilter(predictedYaw, windowSizes(i));
    figureHandle = figure;
    plot(expectedYaw, 'g', 'LineWidth', 2);
    hold on;
    plot(filteredYaw, 'r', 'LineWidth', 2);
    hold off;
    legend({'True', sprintf('Filtered (windowsize: %d)', windowSizes(i))});
    xlabel(sprintf('Frame number/Time'));
    ylabel('Azimuth bin number (total 21 bins)');
    ylim([0 25]);
    title(sprintf('Consistency of Azimuth prediction over frames \n Car ID %02d, seq %02d, frames: %03d-%03d \n Number of accurate predictions: %03d/%03d', infoStruct.carId, infoStruct.seqNum, infoStruct.firstFrameId, infoStruct.lastFrameId, sum(expectedYaw == filteredYaw), length(expectedYaw)));

    if saveConsistencyPlots
        saveas(figureHandle, sprintf('%s/seq%2d_car%2d_medianFilter_windowSize%02d.jpg', vpConsistencyResultsdir, infoStruct.seqNum, infoStruct.carId, windowSizes(i)));
    end

end

% Close all windows
close all;

end