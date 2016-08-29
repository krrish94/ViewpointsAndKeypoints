function kpsPerm = findKpsPerm(part_names)
% FINDKPSPERM  Finds the permutation of keypoint indices that should be
% assigned, when training samples are flipped horizontally.

% Get all indices starting with 'left'
leftInds = cellfun(@(x) ~isempty(x),strfind(part_names,'Left'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(part_names,'L_'));
leftInds = leftInds | cellfun(@(x) ~isempty(x),strfind(part_names,'left'));

% Get all indices starting with 'right'
rightInds = cellfun(@(x) ~isempty(x),strfind(part_names,'Right'));
rightInds = rightInds | cellfun(@(x) ~isempty(x),strfind(part_names,'R_'));
rightInds = rightInds | cellfun(@(x) ~isempty(x),strfind(part_names,'right'));

% Initialize flipNames with part_names
flipNames = part_names;

% Right becomes left, and left becomes right
flipNames(leftInds) = strrep(flipNames(leftInds),'L_','R_');
flipNames(leftInds) = strrep(flipNames(leftInds),'Left','Right');
flipNames(leftInds) = strrep(flipNames(leftInds),'left','right');

flipNames(rightInds) = strrep(flipNames(rightInds),'R_','L_');
flipNames(rightInds) = strrep(flipNames(rightInds),'Right','Left');
flipNames(rightInds) = strrep(flipNames(rightInds),'right','left');

kpsPerm = zeros(length(flipNames),1);
for i=1:length(flipNames)
    kpsPerm(i) = find(ismember(part_names,flipNames{i}));
end

end

