import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import myenv
from myagent import SarsaLambdaAgent, RandomAgent

def evaluate_greedy(
    agent: SarsaLambdaAgent,
    env: gym.Env,
    episodes: int = 100,
) -> tuple[list[float], list[int]]:
    #this is just for debugging, not using this in the actual plots

    returns = []
    final_infected_counts = []
    

    for _ in range(episodes):
        observation, _ = env.reset()
        total_return = 0.0

        while True:
            action = agent.eps_greedy(
                observation,
                exploration=False,
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_return += reward

            if terminated or truncated:
                break

        returns.append(total_return)
        final_infected_counts.append(info["infected_count"])

    return returns, final_infected_counts

def run_lambda_sweep(
    lambdas: list[float],
    seeds: list[int],
    episodes: int = 5000,
) -> dict[float, np.ndarray]:
    """Train one agent for every lambda/seed combination."""
    results = {}

    for lam in lambdas:
        seed_returns = []
        for seed in seeds:
            env = gym.make("cs272/MyEnv-v0")
            env.reset(seed=seed)
            agent = SarsaLambdaAgent(
                env,
                gamma=0.99,
                alpha=0.05,
                eps=0.1,
                lam=lam,
                total_epi=episodes,
                init_val=1.0,
                seed=seed,
            )

            returns = agent.learn()
            seed_returns.append(returns)

            env.close()

        results[lam] = np.array(seed_returns)

    return results

def run_random_baseline(
    seeds: list[int],
    episodes: int = 5_000,
) -> np.ndarray:
    """Train the provided random agent for each seed."""
    seed_returns = []

    for seed in seeds:
        env = gym.make("cs272/MyEnv-v0")
        env.reset(seed=seed)

        agent = RandomAgent(
            env,
            total_epi=episodes,
            seed=seed,
        )

        returns = agent.learn()
        seed_returns.append(returns)

        env.close()

    return np.array(seed_returns)


def plot_lambda_sweep(
    sweep_results: dict[float, np.ndarray],
    random_returns: np.ndarray,
    window: int = 100,
    output_path: str = "lambda_sweep.png",
) -> None:
    """Plot moving-average returns for each lambda and random play."""
    plt.figure(figsize=(12, 6))

    kernel = np.ones(window) / window

    for lam in sorted(sweep_results):
        returns = sweep_results[lam]

        # Smooth each seed independently.
        smoothed_returns = np.array([
            np.convolve(
                seed_returns,
                kernel,
                mode="valid",
            )
            for seed_returns in returns
        ])

        mean_returns = smoothed_returns.mean(axis=0)
        std_returns = smoothed_returns.std(axis=0)

        episodes = np.arange(
            window,
            returns.shape[1] + 1,
        )

        plt.plot(
            episodes,
            mean_returns,
            label=f"lambda={lam}",
        )

        plt.fill_between(
            episodes,
            mean_returns - std_returns,
            mean_returns + std_returns,
            alpha=0.12,
        )

    # Smooth and plot the random-agent baseline.
    random_smoothed = np.array([
        np.convolve(
            seed_returns,
            kernel,
            mode="valid",
        )
        for seed_returns in random_returns
    ])

    random_mean = random_smoothed.mean(axis=0)
    random_std = random_smoothed.std(axis=0)

    episodes = np.arange(
        window,
        random_returns.shape[1] + 1,
    )

    plt.plot(
        episodes,
        random_mean,
        color="black",
        linestyle="--",
        linewidth=2,
        label="random agent",
    )

    plt.fill_between(
        episodes,
        random_mean - random_std,
        random_mean + random_std,
        color="gray",
        alpha=0.15,
    )

    plt.xlabel("Episode")
    plt.ylabel("Mean return")
    plt.title(
        f"{window}-Episode Moving Average: "
        "SARSA(lambda) vs Random Agent"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

def summarize_lambda_results(
    sweep_results: dict[float, np.ndarray],
    target_return: float = 2.0,
    window: int = 100,
) -> None:
    print("\nlambda results")
    print("-" * 60)
    print(
        f"{'ambda':<12}"
        f"{'first episode above threshold':<35}"
        f"{'mean final return':<20}"
    )
    print("-" * 60)

    for lam in sorted(sweep_results):
        returns = sweep_results[lam]

        smoothed = np.array([
            np.convolve(
                seed_returns,
                np.ones(window) / window,
                mode="valid",
            )
            for seed_returns in returns
        ])

        mean_curve = smoothed.mean(axis=0)

        reached = np.flatnonzero(
            mean_curve >= target_return
        )

        if len(reached) == 0:
            first_episode = "never"
        else:
            first_episode = int(reached[0] + window)

        final_return = float(mean_curve[-1])

        print(
            f"{lam:<12}"
            f"{str(first_episode):<35}"
            f"{final_return:.3f}"
        )

    print(f"\nTarget return threshold: {target_return}")
    print(f"Moving-average window: {window}")

def print_experiment_parameters(
    lambdas: list[float],
    seeds: list[int],
    episodes: int,
    window: int,
    target_return: float,
) -> None:
    parameters = {
        "Environment": "cs272/MyEnv-v0",
        "People": 6,
        "Episodes per run": episodes,
        "Lambda values": lambdas,
        "Seeds": seeds,
        "Gamma": 0.99,
        "Alpha": 0.05,
        "Epsilon": 0.15,
        "Initial Q value": 1.0,
        "Trace type": "accumulating",
        "Moving-average window": window,
        "Target return": target_return,
    }

    print("\nreproducibility parameters")
    print("-" * 45)

    for name, value in parameters.items():
        print(f"{name:<25}: {value}")

def show_greedy_episode(
    agent: SarsaLambdaAgent,
    env: gym.Env,
    max_steps: int = 300,
) -> None:
    observation, _ = env.reset(seed=0)
    total_return = 0.0

    print("\nfully greedy policy episode")
    print(env.render())

    for step in range(max_steps):
        action = agent.eps_greedy(
            observation,
            exploration=False,
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_return += reward

        print(f"\nStep {step + 1}")
        print(f"Observation: {observation}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(env.render())

        observation = next_observation

        if terminated or truncated:
            break

    print(f"\nFinal return: {total_return}")
    print(f"Final infected count: {info['infected_count']}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")


def main():
    env = gym.make(
        "cs272/MyEnv-v0",
        render_mode="ansi",
    )

    env.reset(seed=0)
    print("Initial environment:")
    print(env.render())

    agent = SarsaLambdaAgent(
        env,
        gamma=0.99,
        alpha=0.05,
        eps=0.1,
        lam=0.9,
        total_epi=5000,
        init_val=1.0,
        seed=0,
    )

    training_returns = agent.learn()

    print("\nTraining complete")
    # This is a VERY stochastic environment, so episodic returns arent that meaningful
    # so I'm using moving averages
    print(
        "First 100 average:",
        np.mean(training_returns[:100]),
    )
    print(
        "Last 100 average:",
        np.mean(training_returns[-100:]),
    )

    # Evaluate the learned policy greedily.
    greedy_returns, infected_counts = evaluate_greedy(
        agent,
        env,
        episodes=100,
    )
    
    print(
        "Greedy average return:",
        np.mean(greedy_returns),
    )
    
    print(
        "Average final infected count:",
        np.mean(infected_counts),
    )
    
    print(
        "Final infected-count distribution:",
        {
            count: infected_counts.count(count)
            for count in sorted(set(infected_counts))
        },
    )
    env.close()


    episode, reached_terminal = agent.best_run()
    
    print("greedy episode:", episode)
    print("rreached terminal:", reached_terminal)
    print("undiscounted return:", agent.calc_return(episode))
    # print("discounted return:", agent.calc_return(episode, discounted=True))


    lambdas = [0.0, 0.3, 0.6, 0.9, 1.0]
    seeds = [0, 1, 2, 3, 4]
    
    sweep_results = run_lambda_sweep(
        lambdas=lambdas,
        seeds=seeds,
        episodes=5000,
    )

    random_returns = run_random_baseline(
        seeds=seeds,
        episodes=5000,
    )


    target_return = 2.0
    window = 100
    
    plot_lambda_sweep(
         sweep_results,
         random_returns,
         window=window,
     )
    

    # for lam, returns in sweep_results.items():
    #     print(
    #         f"lambda={lam}: "
    #         f"first100={returns[:, :500].mean():.3f}, "
    #         f"last100={returns[:, -500:].mean():.3f}"
    #     )
    #
    # print(
    #     "random:",
    #     f"first100={random_returns[:, :500].mean():.3f}, "
    #     f"last100={random_returns[:, -500:].mean():.3f}",
    # )
    #
    print_experiment_parameters(
        lambdas=lambdas,
        seeds=seeds,
        episodes=5000,
        window=window,
        target_return=target_return,
    )
    
    summarize_lambda_results(
        sweep_results,
        target_return=target_return,
        window=window,
    )





    report_env = gym.make(
        "cs272/MyEnv-v0",
        render_mode="ansi",
    )
    
    report_env.reset(seed=0)
    
    report_agent = SarsaLambdaAgent(
        report_env,
        gamma=0.99,
        alpha=0.05,
        eps=0.15,
        lam=0.6,
        total_epi=5000,
        init_val=1.0,
        seed=0,
    )
    
    report_agent.learn()
    show_greedy_episode(report_agent, report_env)
    report_env.close()

if __name__ == "__main__":
    main()
