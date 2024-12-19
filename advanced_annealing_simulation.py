from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Callable, Optional
import math
import random
import time
from dataclasses import dataclass
from contextlib import contextmanager
import logging
from enum import Enum
import numpy as np

# Type variables for generic implementations
T = TypeVar('T')  # Solution type
C = TypeVar('C')  # Cost type

class CoolingSchedule(Enum):
    """Enumeration of available cooling schedule strategies."""
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    QUADRATIC = "quadratic"
    ADAPTIVE = "adaptive"

@dataclass
class SimulationStatistics:
    """Statistics collector for the annealing process."""
    iterations: int = 0
    accepted_moves: int = 0
    rejected_moves: int = 0
    temperature_history: List[float] = None
    cost_history: List[float] = None
    best_cost: float = float('inf')
    
    def __post_init__(self):
        self.temperature_history = []
        self.cost_history = []

    def acceptance_rate(self) -> float:
        """Calculate the acceptance rate of proposed moves."""
        total_moves = self.accepted_moves + self.rejected_moves
        return self.accepted_moves / total_moves if total_moves > 0 else 0.0

class Observer(ABC):
    """Abstract Observer for the Observer pattern."""
    @abstractmethod
    def update(self, temperature: float, current_cost: float, best_cost: float) -> None:
        """Update observer with current simulation state."""
        pass

class SimulationLogger(Observer):
    """Concrete Observer implementing logging functionality."""
    def __init__(self, log_interval: int = 100):
        self.logger = logging.getLogger(__name__)
        self.log_interval = log_interval
        self.iteration = 0

    def update(self, temperature: float, current_cost: float, best_cost: float) -> None:
        """Log simulation progress at specified intervals."""
        self.iteration += 1
        if self.iteration % self.log_interval == 0:
            self.logger.info(
                f"Iteration {self.iteration}: T={temperature:.4f}, "
                f"Current Cost={current_cost:.4f}, Best Cost={best_cost:.4f}"
            )

class CoolingScheduleStrategy(ABC):
    """Strategy pattern for cooling schedule implementations."""
    @abstractmethod
    def calculate_temperature(self, initial_temp: float, current_iteration: int, 
                            max_iterations: int) -> float:
        """Calculate current temperature based on iteration progress."""
        pass

class ExponentialCooling(CoolingScheduleStrategy):
    """Exponential cooling schedule implementation."""
    def __init__(self, alpha: float = 0.95):
        self.alpha = alpha

    def calculate_temperature(self, initial_temp: float, current_iteration: int,
                            max_iterations: int) -> float:
        """Implement exponential cooling schedule."""
        return initial_temp * (self.alpha ** current_iteration)

class AdaptiveCooling(CoolingScheduleStrategy):
    """Adaptive cooling schedule using acceptance rate feedback."""
    def __init__(self, target_acceptance_rate: float = 0.44):
        self.target_acceptance_rate = target_acceptance_rate
        self.acceptance_history: List[bool] = []
        self.window_size = 100

    def calculate_temperature(self, initial_temp: float, current_iteration: int,
                            max_iterations: int) -> float:
        """Implement adaptive cooling based on acceptance history."""
        if not self.acceptance_history:
            return initial_temp
            
        current_acceptance_rate = sum(self.acceptance_history[-self.window_size:]) / \
                                min(len(self.acceptance_history), self.window_size)
        
        adjustment = 1.0 + 0.1 * (current_acceptance_rate - self.target_acceptance_rate)
        return initial_temp * (adjustment ** current_iteration)

class SolutionFactory(Generic[T], ABC):
    """Abstract Factory pattern for creating and manipulating solutions."""
    @abstractmethod
    def create_initial_solution(self) -> T:
        """Create an initial solution for the problem."""
        pass

    @abstractmethod
    def create_neighbor(self, current_solution: T) -> T:
        """Generate a neighbor solution from the current one."""
        pass

    @abstractmethod
    def calculate_cost(self, solution: T) -> float:
        """Calculate the cost of a given solution."""
        pass

class SimulatedAnnealing(Generic[T]):
    """Main Simulated Annealing implementation using Template Method pattern."""
    def __init__(
        self,
        solution_factory: SolutionFactory[T],
        initial_temperature: float,
        min_temperature: float,
        max_iterations: int,
        cooling_schedule: CoolingSchedule = CoolingSchedule.EXPONENTIAL
    ):
        self.solution_factory = solution_factory
        self.initial_temperature = initial_temperature
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.statistics = SimulationStatistics()
        self.observers: List[Observer] = []
        
        # Strategy pattern for cooling schedule
        self.cooling_schedule: CoolingScheduleStrategy = self._create_cooling_schedule(cooling_schedule)

    def _create_cooling_schedule(self, schedule_type: CoolingSchedule) -> CoolingScheduleStrategy:
        """Factory method for creating cooling schedule strategies."""
        if schedule_type == CoolingSchedule.EXPONENTIAL:
            return ExponentialCooling()
        elif schedule_type == CoolingSchedule.ADAPTIVE:
            return AdaptiveCooling()
        else:
            raise ValueError(f"Unsupported cooling schedule: {schedule_type}")

    def add_observer(self, observer: Observer) -> None:
        """Add an observer to the simulation."""
        self.observers.append(observer)

    def _notify_observers(self, temperature: float, current_cost: float, best_cost: float) -> None:
        """Notify all observers of the current state."""
        for observer in self.observers:
            observer.update(temperature, current_cost, best_cost)

    @contextmanager
    def _performance_monitoring(self):
        """Context manager for monitoring performance metrics."""
        start_time = time.time()
        yield
        execution_time = time.time() - start_time
        logging.info(f"Execution time: {execution_time:.2f} seconds")
        logging.info(f"Final acceptance rate: {self.statistics.acceptance_rate():.2%}")

    def _acceptance_probability(self, current_cost: float, new_cost: float, 
                              temperature: float) -> float:
        """Calculate probability of accepting a worse solution."""
        if new_cost < current_cost:
            return 1.0
        return math.exp((current_cost - new_cost) / temperature)

    def optimize(self) -> tuple[T, float]:
        """Main optimization loop implementing the simulated annealing algorithm."""
        with self._performance_monitoring():
            current_solution = self.solution_factory.create_initial_solution()
            current_cost = self.solution_factory.calculate_cost(current_solution)
            best_solution = current_solution
            best_cost = current_cost
            
            temperature = self.initial_temperature
            
            while (temperature > self.min_temperature and 
                   self.statistics.iterations < self.max_iterations):
                
                # Generate and evaluate neighbor solution
                neighbor = self.solution_factory.create_neighbor(current_solution)
                neighbor_cost = self.solution_factory.calculate_cost(neighbor)
                
                # Calculate acceptance probability
                acceptance_prob = self._acceptance_probability(
                    current_cost, neighbor_cost, temperature
                )
                
                # Accept or reject the neighbor solution
                if random.random() < acceptance_prob:
                    current_solution = neighbor
                    current_cost = neighbor_cost
                    self.statistics.accepted_moves += 1
                    
                    # Update best solution if necessary
                    if current_cost < best_cost:
                        best_solution = current_solution
                        best_cost = current_cost
                else:
                    self.statistics.rejected_moves += 1
                
                # Update statistics and notify observers
                self.statistics.iterations += 1
                self.statistics.temperature_history.append(temperature)
                self.statistics.cost_history.append(current_cost)
                self.statistics.best_cost = best_cost
                
                self._notify_observers(temperature, current_cost, best_cost)
                
                # Update temperature according to cooling schedule
                temperature = self.cooling_schedule.calculate_temperature(
                    self.initial_temperature,
                    self.statistics.iterations,
                    self.max_iterations
                )
        
        return best_solution, best_cost

# Example implementation for solving the Traveling Salesman Problem
class TSPSolution:
    """Represents a solution for the Traveling Salesman Problem."""
    def __init__(self, route: List[int]):
        self.route = route

    def __str__(self) -> str:
        return f"Route: {' -> '.join(map(str, self.route))}"

class TSPSolutionFactory(SolutionFactory[TSPSolution]):
    """Concrete factory for TSP solutions."""
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.size = len(distance_matrix)

    def create_initial_solution(self) -> TSPSolution:
        """Create initial random route."""
        route = list(range(self.size))
        random.shuffle(route)
        return TSPSolution(route)

    def create_neighbor(self, current_solution: TSPSolution) -> TSPSolution:
        """Generate neighbor solution using 2-opt move."""
        new_route = current_solution.route.copy()
        i, j = sorted(random.sample(range(self.size), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return TSPSolution(new_route)

    def calculate_cost(self, solution: TSPSolution) -> float:
        """Calculate total route distance."""
        route = solution.route
        return sum(self.distance_matrix[route[i-1]][route[i]]
                  for i in range(len(route)))

# Example usage
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create sample TSP problem
    size = 20
    distance_matrix = np.random.rand(size, size)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(distance_matrix, 0)  # Zero diagonal

    # Create solution factory
    factory = TSPSolutionFactory(distance_matrix)

    # Create and configure simulated annealing solver
    solver = SimulatedAnnealing(
        solution_factory=factory,
        initial_temperature=100.0,
        min_temperature=0.01,
        max_iterations=10000,
        cooling_schedule=CoolingSchedule.ADAPTIVE
    )

    # Add observers
    solver.add_observer(SimulationLogger(log_interval=500))

    # Run optimization
    best_solution, best_cost = solver.optimize()

    # Print results
    print(f"\nBest solution found: {best_solution}")
    print(f"Best cost: {best_cost:.4f}")
    print(f"Total iterations: {solver.statistics.iterations}")
    print(f"Acceptance rate: {solver.statistics.acceptance_rate():.2%}")

if __name__ == "__main__":
    main()