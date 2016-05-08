function [testErrors,testMedErrors,testErrs,testData,testPreds,testLabels] = regressToPose(class)

%[testErrors,testMedErrors] = regressToPose(class)
%   uses the training/val/test sets specified in parameters and
% regresses to pose and returns error


% Declare global variables
globals;
% Encoding of the output angle (default - Euler)
encoding = params.angleEncoding;

% Create train and test (val) sets - enough if called once per class
createEvalSets(class);

% Loading the train/test features and labels for the current class
data = load(fullfile(cachedir,'evalSets',class));
[trainLabels,testLabels,trainFeats,testFeats] = generateEvalSetData(data);

% Obtain predictions on train and test data
switch params.optMethod
    % Currently the only technique
    case 'bin'
        % Weights to be assigned to mirrored features
        alphaOpt = 0;
        % Number of hypothesis to consider (usually 1, maximum of 8
        % hypotheses supported currently)
        nHypotheses = params.nHypotheses;
        
        if strcmp(params.features, 'vggAzimuthVps')
            testPreds = km_poseHypotheses(testFeats, nHypotheses, alphaOpt);
            trainPreds = km_poseHypotheses(trainFeats, nHypotheses, alphaOpt);
        else
            % Predict pose on test set
            [testPreds] = poseHypotheses(testFeats,nHypotheses,alphaOpt);
            % Predict pose on train set
            [trainPreds] = poseHypotheses(trainFeats,nHypotheses,alphaOpt); 
        end
end

% Evaluate test predictions and compute test error
testErrs = evaluatePredictionError(testPreds,testLabels,encoding);
% Evaluate train predictions and compute train error
trainErrs = evaluatePredictionError(trainPreds,trainLabels,encoding);
% [mean(trainErrs), mean(testErrs)]
% [median(trainErrs) median(testErrs)]

%diff = testPreds - testLabels;
%mean(sum(diff.*diff,2));

% Arrays to store mean and median of the test error
testErrors = [];
testMedErrors=[];

% Store mean of test error
testErrors(1) = mean(testErrors);
% Store median of test error
testMedErrors(1) = median(testErrors);

% Compute 'binned' error (bins of size 30 degrees)
testErrors(1) = sum(testErrs<=30)/numel(testErrs);

% Looks lika a repeat computation (TODO: check and delete)
testMedErrors(1) = median(testErrs);

% Store test data (to return)
testData = data.test;

% Sort test errors
[errSort,IDX] = sort(testErrs,'ascend');

end
