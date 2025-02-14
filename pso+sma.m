% Load Features (Extracted from CNNs)
features_combined = [features_squeeze, features_inception];  % Fused Features from SqueezeNet & InceptionResNet-V2

% Parameters
numParticles = 20;   % Number of Particles for PSO
maxIterations = 50;  % Maximum Iterations

% Apply PSO for Feature Optimization
optimized_features_pso = pso_feature_selection(features_combined, labels, numParticles, maxIterations);


% Parameters
numAgents = 20;      % Number of Agents for SMA
maxIterations = 50;  % Maximum Iterations

% Apply SMA for Feature Optimization
optimized_features_sma = sma_feature_selection(features_combined, labels, numAgents, maxIterations);


% Split Data for Training and Testing (for PSO)
numTrain = floor(0.8 * size(optimized_features_pso, 1));
trainData_pso = optimized_features_pso(1:numTrain, :);
trainLabels_pso = labels(1:numTrain);
testData_pso = optimized_features_pso(numTrain+1:end, :);
testLabels_pso = labels(numTrain+1:end);

% Train SVM Classifier (for PSO)
svmModel_pso = fitcsvm(trainData_pso, trainLabels_pso, 'KernelFunction', 'linear');
predicted_pso = predict(svmModel_pso, testData_pso);
accuracy_pso = sum(predicted_pso == testLabels_pso) / numel(testLabels_pso) * 100;

fprintf('Classification Accuracy with PSO-Optimized Features: %.2f%%\n', accuracy_pso);

% Split Data for Training and Testing (for SMA)
numTrain = floor(0.8 * size(optimized_features_sma, 1));
trainData_sma = optimized_features_sma(1:numTrain, :);
trainLabels_sma = labels(1:numTrain);
testData_sma = optimized_features_sma(numTrain+1:end, :);
testLabels_sma = labels(numTrain+1:end);

% Train SVM Classifier (for SMA)
svmModel_sma = fitcsvm(trainData_sma, trainLabels_sma, 'KernelFunction', 'linear');
predicted_sma = predict(svmModel_sma, testData_sma);
accuracy_sma = sum(predicted_sma == testLabels_sma) / numel(testLabels_sma) * 100;

fprintf('Classification Accuracy with SMA-Optimized Features: %.2f%%\n', accuracy_sma);
