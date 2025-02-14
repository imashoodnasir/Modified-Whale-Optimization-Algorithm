function best_features = pso_feature_selection(features, labels, numParticles, maxIterations)
    % Particle Swarm Optimization for Feature Selection
    numFeatures = size(features, 2);
    
    % Initialize Particles and Velocities
    particles = randi([0, 1], numParticles, numFeatures);
    velocities = zeros(numParticles, numFeatures);
    
    % Initialize Personal and Global Bests
    pBest = particles;
    pBestFitness = zeros(numParticles, 1);
    for i = 1:numParticles
        pBestFitness(i) = evaluate_fitness(features(:, logical(pBest(i, :))), labels);
    end
    
    [gBestFitness, bestIdx] = max(pBestFitness);
    gBest = pBest(bestIdx, :);
    
    % PSO Parameters
    w = 0.7;    % Inertia weight
    c1 = 1.5;   % Cognitive coefficient
    c2 = 1.5;   % Social coefficient
    
    % Main PSO Loop
    for iter = 1:maxIterations
        for i = 1:numParticles
            % Update Velocity
            velocities(i, :) = w * velocities(i, :) + ...
                              c1 * rand * (pBest(i, :) - particles(i, :)) + ...
                              c2 * rand * (gBest - particles(i, :));
            
            % Update Position using Sigmoid
            sigmoid_vel = 1 ./ (1 + exp(-velocities(i, :)));
            particles(i, :) = double(rand(size(sigmoid_vel)) < sigmoid_vel);
            
            % Evaluate New Fitness
            newFitness = evaluate_fitness(features(:, logical(particles(i, :))), labels);
            if newFitness > pBestFitness(i)
                pBest(i, :) = particles(i, :);
                pBestFitness(i) = newFitness;
            end
            
            % Update Global Best
            if newFitness > gBestFitness
                gBest = particles(i, :);
                gBestFitness = newFitness;
            end
        end
    end
    
    % Return Best Features
    best_features = features(:, logical(gBest));
end
