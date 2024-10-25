import numpy as np

def best_response(strategy, payoff_matrix):
    """
    Calculate the best response for a given strategy.
    :param strategy: Current strategy of the opponent.
    :param payoff_matrix: Payoff matrix for the player.
    :return: Best response action index.
    """
    return np.argmax(payoff_matrix @ strategy)

def calculate_nash_equilibrium(payoff_matrix_a, payoff_matrix_b, max_iterations=1000, tolerance=1e-6):
    """
    Find Nash Equilibrium using best response dynamics.
    :param payoff_matrix_a: Payoff matrix for player A.
    :param payoff_matrix_b: Payoff matrix for player B.
    :param max_iterations: Maximum number of iterations to run.
    :param tolerance: Tolerance level for convergence.
    :return: Nash Equilibrium strategies for both players.
    """
    num_strategies_a = payoff_matrix_a.shape[0]
    num_strategies_b = payoff_matrix_b.shape[0]

    # Initialize uniform strategies
    strategy_a = np.ones(num_strategies_a) / num_strategies_a
    strategy_b = np.ones(num_strategies_b) / num_strategies_b

    for _ in range(max_iterations):
        # Store previous strategies for convergence check
        prev_strategy_a = strategy_a.copy()
        prev_strategy_b = strategy_b.copy()

        # Update strategies based on best responses
        best_a = best_response(strategy_b, payoff_matrix_a)
        strategy_a = np.zeros(num_strategies_a)
        strategy_a[best_a] = 1

        best_b = best_response(strategy_a, payoff_matrix_b)
        strategy_b = np.zeros(num_strategies_b)
        strategy_b[best_b] = 1

        # Check for convergence
        if np.allclose(prev_strategy_a, strategy_a, atol=tolerance) and \
           np.allclose(prev_strategy_b, strategy_b, atol=tolerance):
            break

    return strategy_a, strategy_b

def main():
    # Example payoff matrices for two players (2x2 game)
    try:
        payoff_matrix_a = np.array([[3, 0], [5, 1]])  # Player A's payoffs
        payoff_matrix_b = np.array([[3, 5], [0, 1]])  # Player B's payoffs

        # Validate input matrices
        if payoff_matrix_a.shape != payoff_matrix_b.shape:
            raise ValueError("Payoff matrices must have the same dimensions.")

        strategy_a, strategy_b = calculate_nash_equilibrium(payoff_matrix_a, payoff_matrix_b)

        print("Nash Equilibrium Strategies:")
        print("Player A's Strategy:", strategy_a)
        print("Player B's Strategy:", strategy_b)

    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
