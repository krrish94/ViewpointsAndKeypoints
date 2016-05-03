% disp('Loading conv6')
% 
% class = pascalIndexClass(c);
% load(fullfile(cachedir,'rcnnPredsKps', [params.kpsNet 'Conv6Kps'],class));
% feat = flipMapXY(feat,[6 6]);
% feat6 = resizeHeatMap(feat,[6 6]);
% featConv6 = 1./(1+exp(-feat6));
% feat=featConv6;
% 
% disp('Loading conv12')
% 
% load(fullfile(cachedir,'rcnnPredsKps', [params.kpsNet 'Conv12Kps'],class));
% feat = flipMapXY(feat,[12 12]);
% feat12 = resizeHeatMap(feat,[12 12]);
% featConv6 = 1./(1+exp(-feat12));
% feat=1./(1+exp(-feat12-feat6));
% 
% disp('Predicting keypoints');

% Read in image and get the bounding box
imName = '000000.png';
bbox = [295, 170, 462, 290];

imName = '000001.png';
bbox = [295, 170, 462, 290];


im = imread(fullfile(basedir, 'data', 'KITTI', 'Seq00', imName));
% Predict keypoints and their scores
[kpCoords,scores] = maxLocationPredict(feat(i,:),bbox,hDims);
% Get the first 14 keypoints
kpCoords = kpCoords(1:2,1:14);

bbox2(1) = bbox(1); bbox2(2) = bbox(2); bbox2(3) = bbox(3)-bbox(1); bbox2(4) = bbox(4)-bbox(2);
imshow(im);
hold on
scatter(kpCoords(1,:),kpCoords(2,:),50,'r','filled')
scatter(kps(:,1),kps(:,2),50,'b','filled')
rectangle('Position', bbox2, 'LineWidth', 3, 'EdgeColor', 'g');
hold off














