# Cascading CMA-ES Instances for Generating Input-diverse Solution Batches

This repository contains the implementation of the CMA-ES Diversity Search algorithm, experimental results, and log files generated during execution.

# Repository Structure
-images/: Contains plots of the results obtained from experiments.
-cma_es_ds.py: The main Python script implementing the CMA-ES Diversity Search algorithm.
-log/: Stores log files generated during execution.
-requirements.txt: List of required dependencies to run the code.

#Installation

Running this code requires Python 3.10, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```
#Running the Algorithm
Execute the script with:
```
python cma_es_ds.py
```
#Parameters
The script allows customization through the following parameters:
-dimension (int): Dimensionality of the optimization problem.
-maxfevals (int): Maximum number of function evaluations.
-dmin_list (list of int): List of minimum distance constraints.
-problem_list (list of int): List of optimization problems to solve.
-batch_size (int): Number of solutions inside the final batch.
Modify these parameters directly in cma_es_ds.py before execution.

#Log Files
For each combination of function and distance, the algorithm produces three log files:
1. Sample File (log/sample_*.txt)
 -Contains all evaluated points.
 -Column 0: Loss value of the point.
 -Subsequent columns: Coordinates of the evaluated points.
2. Progression File (log/progression_*.txt)
 -Tracks the evolution of the tabu region centers at each iteration (e.g., if dimension = 5 and batch_size = 5, each row contains 25 columns representing the five CMA-ES centers in cascading order).
3. Results File (log/results_*.txt)
 -First batch_size lines: Coordinates of the final batch output points.
 -Next line: Loss values for each point.
 -Last line: Cumulative average loss, as plotted in the paper.
