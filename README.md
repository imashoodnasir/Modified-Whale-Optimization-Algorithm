# Modified Whale Optimization Algorithm (Enhanced WOA)

This repository contains an enhanced version of the **Whale Optimization Algorithm (WOA)** with five key modifications designed to improve optimization performance, especially for **skin cancer classification** and **feature selection** tasks.

## 🚀 Features and Modifications
This implementation includes **five major enhancements** over the standard WOA:

1. **Adaptive Parameter Control** 📉  
   - Uses an **exponential decay function** for better exploration-exploitation balance.
   - Prevents premature convergence and improves global search efficiency.

2. **Lévy Flight Mechanism** 🏃‍♂️  
   - Introduces **long-distance jumps** to escape local optima.
   - Enhances diversity in the search space.

3. **Opposition-Based Learning (OBL)** 🔄  
   - Initializes **opposite solutions** to increase population diversity.
   - Accelerates convergence by selecting better candidates early.

4. **Adaptive Weighting with Chaos Theory** 🎯  
   - Uses **chaotic weighting functions** for controlled randomness.
   - Improves stability during exploitation.

5. **Multi-Objective Pareto Optimization** 🎭  
   - Optimizes **both classification accuracy and feature selection**.
   - Uses **Pareto dominance** to balance multiple objectives.

---

## 🛠 Installation
To use this algorithm, clone the repository:

```bash
git clone https://github.com/imashoodnasir/Modified-Whale-Optimization-Algorithm.git
cd Modified-Whale-Optimization-Algorithm
```

Ensure you have **MATLAB** installed, as this implementation is developed in MATLAB.

---

## 📌 Usage
### 1️⃣ Define Your **Fitness Function**
Modify `fitnessFunction.m` to match your optimization problem:

```matlab
function fitness = fitnessFunction(solution)
    % Example: Skin Cancer Classification
    accuracy = ML_Classifier(solution); % Call your ML model
    numFeatures = sum(solution ~= 0);
    fitness = [1 - accuracy, numFeatures]; % Minimize both classification error and feature count
end
```

### 2️⃣ Run the **Enhanced WOA**
Call the **WOA optimizer** in MATLAB:

```matlab
numWhales = 30; % Population size
maxIter = 100;  % Maximum iterations
dim = 50;       % Number of features (dimensions)
lb = zeros(1, dim); % Lower bound
ub = ones(1, dim);  % Upper bound

[ParetoFront, BestSolution] = Enhanced_WOA(@fitnessFunction, numWhales, maxIter, dim, lb, ub);
```

### 3️⃣ Interpret the **Results**
- **`ParetoFront`**: Returns the **non-dominated solutions** in the Pareto set.
- **`BestSolution`**: The **best compromise solution** optimizing accuracy and feature count.

---

## 📄 License
This project is **open-source** under the **MIT License**.
