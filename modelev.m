% Split Data (Train: 80%, Test: 20%)
numTrain = floor(0.8 * size(selected_features, 1));
trainData = selected_features(1:numTrain, :);
trainLabels = labels(1:numTrain);

testData = selected_features(numTrain+1:end, :);
testLabels = labels(numTrain+1:end);

% Train SVM Classifier
svmModel = fitcsvm(trainData, trainLabels, 'KernelFunction', 'linear');

% Train KNN Classifier
knnModel = fitcknn(trainData, trainLabels, 'NumNeighbors', 5);

% Train Neural Network Classifier
net = patternnet(10);
net = train(net, trainData', trainLabels');

% Predictions
predicted_svm = predict(svmModel, testData);
predicted_knn = predict(knnModel, testData);
predicted_nn = net(testData');

% Calculate Accuracy
accuracy_svm = sum(predicted_svm == testLabels) / numel(testLabels) * 100;
accuracy_knn = sum(predicted_knn == testLabels) / numel(testLabels) * 100;
accuracy_nn = sum(round(predicted_nn') == testLabels) / numel(testLabels) * 100;

% Display Results
fprintf('SVM Accuracy: %.2f%%\n', accuracy_svm);
fprintf('KNN Accuracy: %.2f%%\n', accuracy_knn);
fprintf('Neural Network Accuracy: %.2f%%\n', accuracy_nn);



% Confusion Matrix
confMat_svm = confusionmat(testLabels, predicted_svm);
confMat_knn = confusionmat(testLabels, predicted_knn);
confMat_nn = confusionmat(testLabels, round(predicted_nn'));

% ROC Curve
[X_svm, Y_svm, ~, AUC_svm] = perfcurve(testLabels, predicted_svm, 1);
[X_knn, Y_knn, ~, AUC_knn] = perfcurve(testLabels, predicted_knn, 1);
[X_nn, Y_nn, ~, AUC_nn] = perfcurve(testLabels, round(predicted_nn), 1);

% Plot ROC Curves
figure;
plot(X_svm, Y_svm, '-b', 'LineWidth', 2);
hold on;
plot(X_knn, Y_knn, '-r', 'LineWidth', 2);
plot(X_nn, Y_nn, '-g', 'LineWidth', 2);
legend('SVM', 'KNN', 'Neural Network');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curve for Classification');
grid on;
