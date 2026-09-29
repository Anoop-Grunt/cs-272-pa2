import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import myenv
from myagent import SarsaLambdaAgent, RandomAgent

def describe_observation(observation: int, n_people: int = 6) -> str:
    infected_mask, current_person = divmod(
        observation,
        n_people,
    )

    infected_people = [
        person
        for person in range(n_people)
        if infected_mask & (1 << person)
    ]

    return (
        f"current={current_person}, "
        f"infected={infected_people}"
    )

def gaussian_smooth(values: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)

    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    padded = np.pad(
        values,
        (radius, radius),
        mode="edge",
    )

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )

#
# def evaluate_greedy(
#     agent: SarsaLambdaAgent,
#     env: gym.Env,
#     episodes: int = 100,
# ) -> tuple[list[float], list[int]]:
#     #this is just for debugging, not using this in the actual plots
#     returns = []
#     final_infected_counts = []
#     for _ in range(episodes):
#         observation, _ = env.reset()
#         total_return = 0.0
#
#         while True:
#             action = agent.eps_greedy(
#                 observation,
#                 exploration=False,
#             )
#
#             observation, reward, terminated, truncated, info = (
#                 env.step(action)
#             )
#
#             total_return += reward
#
#             if terminated or truncated:
#                 break
#
#         returns.append(total_return)
#         final_infected_counts.append(info["infected_count"])
#
#     return returns, final_infected_counts
#
def run_lambda_sweep(
    lambdas: list[float],
    seeds: list[int],
    episodes: int = 5000,
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
    return_results = {}
    infected_results = {}

    for lam in lambdas:
        seed_returns = []
        seed_infected_counts = []

        for seed in seeds:
            env = gym.make("cs272/MyEnv-v0")
            env.reset(seed=seed)

            agent = SarsaLambdaAgent(
                env,
                gamma=0.99,
                alpha=0.05,
                eps=0.15,
                lam=lam,
                total_epi=episodes,
                init_val=1.0,
                seed=seed,
            )

            returns = agent.learn()

            infected_counts = [
                info["infected_count"]
                for info in agent.final_infos
            ]

            seed_returns.append(returns)
            seed_infected_counts.append(infected_counts)

            env.close()

        return_results[lam] = np.array(seed_returns)
        infected_results[lam] = np.array(seed_infected_counts)

    return return_results, infected_results

def run_random_baseline(
    seeds: list[int],
    episodes: int = 5000,
) -> np.ndarray:
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

    plt.figure(figsize=(12, 6))
    kernel = np.ones(window) / window
    for curr_lambda in sorted(sweep_results):
        returns = sweep_results[curr_lambda]
        #Smoothing retrns over all the seeds used during the sweeps
        smoothed_returns = np.array([
            np.convolve(
                seed_returns,
                kernel,
                mode="valid",
            )
            for seed_returns in returns
        ])

        mean_returns = gaussian_smooth(
            smoothed_returns.mean(axis=0),
            sigma=8,
        )
        
        std_returns = gaussian_smooth(
            smoothed_returns.std(axis=0),
            sigma=8,
        )

        episodes = np.arange(
            window,
            returns.shape[1] + 1,
        )

        plt.plot(
            episodes,
            mean_returns,
            label=f"lambda={curr_lambda}",
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

    random_mean = gaussian_smooth(
        random_smoothed.mean(axis=0),
        sigma=8,
    )
    
    random_std = gaussian_smooth(
        random_smoothed.std(axis=0),
        sigma=8,
    )
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
    plt.legend(
        fontsize=13,
        title_fontsize=14,
        handlelength=3,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()

def summarize_lambda_results(
    sweep_results: dict[float, np.ndarray],
    infected_results: dict[float, np.ndarray],
    target_return: float = 0.80,
    window: int = 100,
) -> None:
    print("\nlambda results")
    print("-" * 95)
    print(
        f"{'lambda':<12}"
        f"{'first return threshold':<28}"
        f"{'mean final return':<22}"
        f"{'mean final infected':<25}"
    )
    print("-" * 95)

    kernel = np.ones(window) / window

    for lam in sorted(sweep_results):
        returns = sweep_results[lam]
        infected_counts = infected_results[lam]

        # Shape: (number of seeds, number of smoothed episodes)
        smoothed_returns = np.array([
            np.convolve(
                seed_returns,
                kernel,
                mode="valid",
            )
            for seed_returns in returns
        ])

        # Mean learning curve across seeds
        mean_return_curve = smoothed_returns.mean(axis=0)

        reached = np.flatnonzero(
            mean_return_curve >= target_return
        )

        if len(reached) == 0:
            first_episode = "never"
        else:
            # Convert zero-based array index to one-based episode number.
            first_episode = int(reached[0] + window)

        # Mean return during the final moving-average window
        mean_final_return = float(
            returns[:, -window:].mean()
        )

        # Mean infected count during the final moving-average window
        mean_final_infected = float(
            infected_counts[:, -window:].mean()
        )

        print(
            f"{lam:<12}"
            f"{str(first_episode):<28}"
            f"{mean_final_return:<22.3f}"
            f"{mean_final_infected:<25.3f}"
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
        "environment": "cs272/MyEnv-v0",
        "people": 6,
        "episodes per run": episodes,
        "lambda values": lambdas,
        "seeds": seeds,
        "gamma": 0.99,
        "alpha": 0.05,
        "epsilon": 0.15,
        "initial q value": 1.0,
        "trace type": "accumulating",
        "moving-average window": window,
        "target return": target_return,
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

    print("\nfully greedy policy episode in ASCII : DIFFERENT SEED - SO DIFFERNT GRAPH FROM THE BEST RUN EPISODE")
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
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8]
    seeds = [0, 1, 2, 3, 4]
    
    sweep_results, infected_results = run_lambda_sweep(
        lambdas=lambdas,
        seeds=seeds,
        episodes=5000,
    )

    random_returns = run_random_baseline(
        seeds=seeds,
        episodes=5000,
    )


    target_return = 1.0
    window = 100
    
    
    print_experiment_parameters(
        lambdas=lambdas,
        seeds=seeds,
        episodes=5000,
        window=window,
        target_return=target_return,
    )
    
    summarize_lambda_results(
        sweep_results,
        infected_results=infected_results,
        target_return=target_return,
        window=window,
    )

    greedy_episode_env = gym.make(
        "cs272/MyEnv-v0",
        render_mode="ansi",
    )
    
    greedy_episode_env.reset(seed=0)
    
    report_agent = SarsaLambdaAgent(
        greedy_episode_env,
        gamma=0.99,
        alpha=0.05,
        eps=0.15,
        lam=0.6,
        total_epi=5000,
        init_val=1.0,
        seed=0,
    )
    
    report_agent.learn()

    episode, reached_terminal = report_agent.best_run()
    
    print("\nBest run data:")
    for step, (state, action, reward) in enumerate( episode,
        start=1,
    ):
        print(
            f"Step {step}: "
            f"{describe_observation(state)}, "
            f"action={action}, "
            f"reward={reward:.2f}"
        )
    print("Reached terminal:", reached_terminal)
    print(
        "Undiscounted return:",
        report_agent.calc_return(episode),
    )
    print(
        "Discounted return:",
        report_agent.calc_return(
            episode,
            discounted=True,
        ),
    )
    
    show_greedy_episode(report_agent, greedy_episode_env)
    greedy_episode_env.close()


    plot_lambda_sweep(
         sweep_results,
         random_returns,
         window=window,
     )

if __name__ == "__main__":
    main()
