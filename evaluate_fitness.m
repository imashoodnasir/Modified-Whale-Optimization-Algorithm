function accuracy = evaluate_fitness(selected_features, labels)
    % Split Data (80% Train, 20% Test)
    numTrain = floor(0.8 * size(selected_features, 1));
    trainData = selected_features(1:numTrain, :);
    trainLabels = labels(1:numTrain);

    testData = selected_features(numTrain+1:end, :);
    testLabels = labels(numTrain+1:end);

    % Train SVM Model
    model = fitcsvm(trainData, trainLabels, 'KernelFunction', 'linear');
    
    % Test Model
    predictions = predict(model, testData);
    
    % Compute Accuracy
    accuracy = sum(predictions == testLabels) / numel(testLabels);
end
