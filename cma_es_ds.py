import cma
import ioh
import numpy as np
import time

class MaxAttemptsReached(Exception):
    """Custom exception raised when the maximum number of attempts is exceeded."""
    pass

class CmaesDiversitySearch:

    dimension = 2
    algo_name = "CMAdiversity_search"

    def __init__(self, f, dmin, log_file=None, maxfevals=10000):
        """Initialization of the CMA-ES Diversity Search class.

        Args:
            f: Optimization problem to solve.
            dmin: Minimum distance constraint between solutions.
            log_file: Optional log file for saving results.
            maxfevals: Maximum number of function evaluations.
        """
        self.f = f
        self.dmin = dmin
        self.log_file = log_file
        self.maxfevals = maxfevals  # Imposed evaluation budget
        self.nb_fevals = 0          # Counter for function evaluations

        # Logging setup
        algo = self.algo_name
        problem = f.meta_data.problem_id
        base = f"algo_{algo}_problem_{problem}_dmin_{dmin}_maxfevals_{self.maxfevals}.txt"
        self.sample_file = open(f"log/sample_{base}", "w")
        print("# fitness value\tpoint", file=self.sample_file)
        self.results_file = open(f"log/results_{base}", "w")
        print("# 1_batch\tfitness_value\tcumulative_average", file=self.results_file)
        self.progression_file = open(f"log/progression_{base}", "w")

        self.iteration_step = 0
        self.best_points = []  # Store the best points for each restart
        self.x0 = []
        self.es = []
        self.tabu_regions = []
        self.history_points = []
        self.history_points_1 = []
        self.points_history = []  # List of generated points
        self.fitness_history = []  # List of fitness values

    def fitness(self, p):
        """Evaluates the fitness value of a given point.

        Args:
            p: Point to evaluate.

        Returns:
            Fitness value of the point.
        """
        self.nb_fevals += 1
        val = self.f(p) - self.f.optimum.y  # Calculate fitness
        self.points_history.append(list(p))  # Add point to history
        self.fitness_history.append(val)  # Add fitness to history
        print(val, *p, sep='\t', file=self.sample_file)
        return val

    def is_in_tabu_region(self, point, i):
        """Checks if a point is within the tabu regions of previous points.

        Args:
            point: The point to check.
            i: Index of the current point.

        Returns:
            True if the point is in a tabu region, False otherwise.
        """
        if i == 0:  # No tabu regions to check for the first point
            return False

        for prev_index in range(i):
            center = self.tabu_regions[prev_index]
            if np.linalg.norm(point - center) < self.dmin:
                return True
        return False

    def generate_valid_point(self, i, pool_size=1000000):
        """Generates a valid point by sampling from a random pool.

        Args:
            i: Index of the current point.
            pool_size: Number of random points to sample.

        Returns:
            A valid point or None if no valid point is found.
        """
        points_pool = np.random.uniform(self.f.bounds.lb, self.f.bounds.ub, (pool_size, self.dimension))
        np.random.shuffle(points_pool)  # Shuffle the pool for randomness

        for point in points_pool:
            if not self.is_in_tabu_region(point, i):
                return point

        print("No valid point found in the pool.")
        return None

    def solve(self, batch_size, search_time_limit=60):
        """Executes the CMA-ES diversity search algorithm.

        Args:
            batch_size: Number of solutions to generate in each batch.
            search_time_limit: Maximum time allowed for point generation in seconds.
        """
        first_restart = True
        exit_main_loop = False

        while self.nb_fevals <= self.maxfevals:
            print(self.nb_fevals)
            self.bestever = [cma.optimization_tools.BestSolution() for _ in range(batch_size)]
            self.returns_points = [None for _ in range(batch_size)]
            self.returns_fitness = [None for _ in range(batch_size)]

            for i in range(batch_size):
                x0 = self.generate_valid_point(i)
                if x0 is None:
                    if first_restart:
                        print("BREAK: maximum number of attempts reached")
                        raise MaxAttemptsReached
                    else:
                        print("Stopping search after a failed iteration.")
                        exit_main_loop = True
                        break
                else:
                    self.x0.append(x0)
                    self.tabu_regions.append(x0)  # Set initial point as tabu region

            if exit_main_loop:
                break

            for i in range(batch_size):
                self.es.append(cma.CMAEvolutionStrategy(self.x0[i], 1.0, {"bounds": [-5, +5], "seed": 4}))

            active_es = batch_size
            first_iteration = True

            while self.nb_fevals <= self.maxfevals and active_es > 0:
                for i in range(batch_size):
                    if self.es[i].stop():
                        active_es -= 1
                        self.tabu_regions[i] = self.es[i].result.xbest
                        continue

                    population = self.es[i].ask()
                    valid_population = [point for point in population if not self.is_in_tabu_region(point, i)]

                    while len(valid_population) < len(population):
                        new_points = self.es[i].ask()
                        for new_point in new_points:
                            if not self.is_in_tabu_region(new_point, i):
                                valid_population.append(new_point)
                            if len(valid_population) == len(population):
                                break

                        if time.time() - search_time_limit >= search_time_limit:
                            print("BREAK: maximum number of attempts reached")
                            if first_restart and first_iteration:
                                raise MaxAttemptsReached
                            else:
                                self.es[i].x0 = []
                                break

                    pop_scores = [self.fitness(point) for point in valid_population]
                    self.es[i].tell(valid_population, pop_scores)
                    best_index = np.argmin(pop_scores)
                    best_point_in_population = valid_population[best_index]
                    best_score_in_population = pop_scores[best_index]

                    self.tabu_regions[i] = best_point_in_population

                first_iteration = False

            for i in range(batch_size):
                self.history_points.append(self.bestever[i].x)
            for i in range(batch_size):
                self.history_points_1.append(self.returns_points[i])

            first_restart = False
            self.x0 = []
            self.es = []
            self.tabu_regions = []

        selected_points = []
        remaining_indices = list(range(len(self.fitness_history)))

        for _ in range(batch_size):
            if len(remaining_indices) == 0:
                break

            min_fitness_idx = min(remaining_indices, key=lambda idx: self.fitness_history[idx])
            selected_point = self.points_history[min_fitness_idx]
            selected_fitness = self.fitness_history[min_fitness_idx]
            selected_points.append((selected_point, selected_fitness))

            remaining_indices = [
                idx for idx in remaining_indices
                if np.linalg.norm(np.array(self.points_history[idx]) - np.array(selected_point)) >= self.dmin
            ]

        selected_fitness_values = [fitness for _, fitness in selected_points]
        cumulative_sum = 0
        cumulative_averages = []

        for i, fitness in enumerate(selected_fitness_values):
            cumulative_sum += fitness
            cumulative_averages.append(cumulative_sum / (i + 1))

        for point, _ in selected_points:
            print(*point, sep='\t', file=self.results_file)
        for point, _ in selected_points:
            print(*point, end=' ', file=self.progression_file)
        print(file=self.progression_file)
        print(*selected_fitness_values, sep='\t', file=self.results_file)
        print(*cumulative_averages, sep='\t', file=self.results_file)


def main_progression(algo, batch_size=5):
    """Main function to execute the progression of the algorithm across problems.

    Args:
        algo: The algorithm class to execute.
        batch_size: Number of solutions in each batch.
    """
    res = {}
    for pb in range(24):
        res[pb + 1] = []
        for dmin in [1, 3, 5]:
            success = False
            while not success:
                try:
                    print(f"Working on function {pb + 1} with dmin {dmin}")
                    S = CmaesDiversitySearch(ioh.get_problem(pb + 1, 0, CmaesDiversitySearch.dimension), dmin)
                    S.solve(batch_size)
                    success = True
                except MaxAttemptsReached:
                    print(f"Retrying function {pb + 1} with dmin {dmin} due to maximum attempts reached")


if __name__ == "__main__":
    main_progression(CmaesDiversitySearch)

