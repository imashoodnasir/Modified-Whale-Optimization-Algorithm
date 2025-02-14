function best_features = genetic_algorithm(features, labels, popSize, maxGen, mutationRate)
    % Genetic Algorithm for Feature Selection
    % Input:
    %   features - extracted features matrix (samples × features)
    %   labels - corresponding labels for classification
    %   popSize - population size
    %   maxGen - maximum generations
    %   mutationRate - probability of mutation
    % Output:
    %   best_features - optimized feature set
    
    numFeatures = size(features, 2);
    
    % Initialize Population (Binary Mask for Feature Selection)
    population = randi([0, 1], popSize, numFeatures);
    
    for gen = 1:maxGen
        % Evaluate Fitness (Accuracy using SVM)
        fitness = zeros(popSize, 1);
        for i = 1:popSize
            selected_features = features(:, logical(population(i, :)));
            fitness(i) = evaluate_fitness(selected_features, labels);
        end
        
        % Selection (Roulette Wheel Selection)
        probabilities = fitness ./ sum(fitness);
        new_population = population(selection_roulette(probabilities, popSize), :);
        
        % Crossover (Single Point Crossover)
        for i = 1:2:popSize
            if rand < 0.7  % Crossover Probability
                point = randi([1, numFeatures-1]);
                temp = new_population(i, point+1:end);
                new_population(i, point+1:end) = new_population(i+1, point+1:end);
                new_population(i+1, point+1:end) = temp;
            end
        end
        
        % Mutation
        for i = 1:popSize
            if rand < mutationRate
                mutation_point = randi([1, numFeatures]);
                new_population(i, mutation_point) = ~new_population(i, mutation_point);
            end
        end
        
        % Update Population
        population = new_population;
    end
    
    % Return Best Features from the Final Population
    [~, bestIdx] = max(fitness);
    best_features = features(:, logical(population(bestIdx, :)));
end


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


function selected_indices = selection_roulette(probabilities, popSize)
    % Cumulative Probability Distribution
    cumulative_probs = cumsum(probabilities);
    selected_indices = zeros(popSize, 1);
    
    for i = 1:popSize
        r = rand;
        selected_indices(i) = find(cumulative_probs >= r, 1, 'first');
    end
end


% Load Features (Extracted from CNNs)
features_combined = [features_squeeze, features_inception];  % Fused Features from SqueezeNet & InceptionResNet-V2

% Parameters
popSize = 20;       % Population Size
maxGen = 50;        % Number of Generations
mutationRate = 0.05; % Mutation Probability

% Apply Genetic Algorithm for Feature Optimization
optimized_features = genetic_algorithm(features_combined, labels, popSize, maxGen, mutationRate);


% Split Data for Training and Testing
numTrain = floor(0.8 * size(optimized_features, 1));
trainData = optimized_features(1:numTrain, :);
trainLabels = labels(1:numTrain);

testData = optimized_features(numTrain+1:end, :);
testLabels = labels(numTrain+1:end);

% Train SVM Classifier
svmModel = fitcsvm(trainData, trainLabels, 'KernelFunction', 'linear');

% Make Predictions
predicted_labels = predict(svmModel, testData);

% Compute Accuracy
accuracy = sum(predicted_labels == testLabels) / numel(testLabels);
fprintf('Final Classification Accuracy with GA-Optimized Features: %.2f%%\n', accuracy * 100);


