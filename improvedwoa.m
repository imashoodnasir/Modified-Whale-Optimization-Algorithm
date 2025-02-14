function [ParetoFront, BestSolution] = Enhanced_WOA(fitnessFunction, numWhales, maxIter, dim, lb, ub)
    % Enhanced Whale Optimization Algorithm (WOA) with 5 Modifications
    % Inputs:
    % fitnessFunction - Function handle for multi-objective optimization
    % numWhales - Number of whales (population size)
    % maxIter - Maximum iterations
    % dim - Number of features (dimension)
    % lb, ub - Lower and upper bounds of search space

    % Initialize Whale Population
    Positions = lb + (ub - lb) .* rand(numWhales, dim);  
    Fitness = zeros(numWhales, 2); % Store both objectives

    % Compute Initial Fitness & Apply Opposition-Based Learning (OBL)
    OppPositions = lb + ub - Positions;
    for i = 1:numWhales
        Fitness(i, :) = fitnessFunction(Positions(i, :));
        OppFitness = fitnessFunction(OppPositions(i, :));
        if dominates(OppFitness, Fitness(i, :))
            Positions(i, :) = OppPositions(i, :);
            Fitness(i, :) = OppFitness;
        end
    end

    % Pareto Front Initialization
    ParetoFront = [];
    BestSolution = Positions(1, :);
    BestFitness = Fitness(1, :);

    % Main WOA Optimization Loop
    for iter = 1:maxIter
        % Update Adaptive Parameter a (Exponential Decay)
        lambda = 5; 
        a = 2 * exp(-lambda * iter / maxIter);
        
        % Compute Pareto Front Update
        ParetoFront = updatePareto(ParetoFront, Positions, Fitness);
        
        for i = 1:numWhales
            % Select Best Whale from Pareto Front
            idx = randi(size(ParetoFront, 1));
            BestWhale = ParetoFront(idx, 1:dim);

            % Compute Chaotic Adaptive Weight (A)
            r = rand();
            A = 2 * a * sin(2 * pi * r) - a;

            % Lévy Flight for Global Search
            if rand < 0.5
                L = levyFlight(dim);
                Positions(i, :) = Positions(i, :) + 0.01 * L .* (Positions(i, :) - BestWhale);
            else
                % WOA Encircling & Spiral Updating
                if abs(A) < 1
                    Positions(i, :) = BestWhale - A .* abs(BestWhale - Positions(i, :));
                else
                    D = abs(BestWhale - Positions(i, :));
                    Positions(i, :) = BestWhale - A .* D;
                end
            end

            % Apply Bound Constraints
            Positions(i, :) = max(min(Positions(i, :), ub), lb);

            % Compute Fitness
            Fitness(i, :) = fitnessFunction(Positions(i, :));

            % Update Best Solution
            if dominates(Fitness(i, :), BestFitness)
                BestFitness = Fitness(i, :);
                BestSolution = Positions(i, :);
            end
        end
    end
end

% Function to check if solution A dominates B in multi-objective sense
function flag = dominates(A, B)
    flag = all(A <= B) && any(A < B);
end

% Function to update Pareto Front
function ParetoFront = updatePareto(ParetoFront, Positions, Fitness)
    combined = [ParetoFront; [Positions, Fitness]];
    nonDominated = [];
    for i = 1:size(combined, 1)
        isDominated = false;
        for j = 1:size(combined, 1)
            if i ~= j && dominates(combined(j, end-1:end), combined(i, end-1:end))
                isDominated = true;
                break;
            end
        end
        if ~isDominated
            nonDominated = [nonDominated; combined(i, :)];
        end
    end
    ParetoFront = nonDominated;
end

% Lévy Flight Mechanism
function L = levyFlight(dim)
    beta = 1.5;
    sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, dim) * sigma;
    v = randn(1, dim);
    L = u ./ abs(v).^(1 / beta);
end


function fitness = fitnessFunction(solution)
    % Example: Skin Cancer Classification
    % Compute classification error (1 - accuracy) and number of selected features
    accuracy = ML_Classifier(solution); % Call your machine learning model
    numFeatures = sum(solution ~= 0);
    fitness = [1 - accuracy, numFeatures]; % Minimize both
end


numWhales = 30;
maxIter = 100;
dim = 50; % Example: Number of features
lb = zeros(1, dim);
ub = ones(1, dim);
[ParetoFront, BestSolution] = Enhanced_WOA(@fitnessFunction, numWhales, maxIter, dim, lb, ub);
