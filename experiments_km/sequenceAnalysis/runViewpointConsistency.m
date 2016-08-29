%% Script to get viewpoint consistency over a batch of sequences and cars


% Get sequence numbers and car ids that are visible for more than a
% threshold number of frames.
disp('Getting sequence numbers and car ids to work on');
minFrames = 70;
seqAndCarIds = getSequencesWithLongTracks(minFrames);

% For each of the cars, run the viewpoint consistency script and save
% results.
for i = 1:size(seqAndCarIds,1)
    sequenceNum = seqAndCarIds(i,1);
    carIds = [seqAndCarIds(i,2)];
    viewpointConsistency;
end