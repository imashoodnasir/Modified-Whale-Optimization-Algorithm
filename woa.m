function best_features = WOA(features, maxIterations, populationSize)
    % Initialize parameters
    a = 2; % Exploration-exploitation balance
    b = 1.5; % Weighting factor
    lb = min(features(:)); % Lower bound
    ub = max(features(:)); % Upper bound

    % Initialize whale positions
    whales = lb + (ub - lb) * rand(populationSize, size(features, 2));
    
    % Main WOA loop
    for iter = 1:maxIterations
        % Update coefficients
        a = 2 - (2 * iter / maxIterations);
        A = 2 * a * rand - a;
        C = 2 * rand;

        % Find best whale
        [~, best_idx] = min(sum(whales.^2, 2));
        best_whale = whales(best_idx, :);

        % Update whale positions
        for i = 1:populationSize
            if rand < 0.5
                D = abs(C * best_whale - whales(i, :));
                whales(i, :) = best_whale - A * D;
            else
                r = rand(size(whales(i, :)));
                whales(i, :) = whales(i, :) + r .* (ub - lb);
            end
        end
    end
    best_features = best_whale;
end

% Parameters
maxIterations = 50;
populationSize = 30;

% Feature Selection
selected_features = WOA(features_combined, maxIterations, populationSize);
