function best_features = sma_feature_selection(features, labels, numAgents, maxIterations)
    % Slime Mould Algorithm for Feature Selection
    numFeatures = size(features, 2);
    
    % Initialize Slime Mould Population
    agents = randi([0, 1], numAgents, numFeatures);
    fitness = zeros(numAgents, 1);
    
    % Compute Initial Fitness
    for i = 1:numAgents
        fitness(i) = evaluate_fitness(features(:, logical(agents(i, :))), labels);
    end
    
    % Find Best Solution
    [bestFitness, bestIdx] = max(fitness);
    bestSolution = agents(bestIdx, :);
    
    % SMA Parameters
    a = 1.5;  % Constant for exploration-exploitation
    b = 1.5;  % Weighting coefficient
    
    % Main SMA Loop
    for iter = 1:maxIterations
        % Compute Probability of Movement
        P = exp(-fitness / max(fitness));
        
        for i = 1:numAgents
            if rand < P(i)  % Explore new positions
                agents(i, :) = randi([0, 1], 1, numFeatures);
            else  % Update position based on best slime mould
                r = rand;
                if r < 0.5
                    agents(i, :) = bestSolution + a * (randn(1, numFeatures));
                else
                    agents(i, :) = bestSolution - b * (randn(1, numFeatures));
                end
                agents(i, agents(i, :) > 1) = 1;
                agents(i, agents(i, :) < 0) = 0;
            end
            
            % Evaluate New Fitness
            newFitness = evaluate_fitness(features(:, logical(agents(i, :))), labels);
            if newFitness > fitness(i)
                fitness(i) = newFitness;
                if newFitness > bestFitness
                    bestSolution = agents(i, :);
                    bestFitness = newFitness;
                end
            end
        end
    end
    
    % Return Best Features
    best_features = features(:, logical(bestSolution));
end
