import gymnasium as gym
import myenv  # registers your custom environment


def main():
    env = gym.make(
        "cs272/VirusSpread-v0",
        render_mode="ansi",
    )

    observation, info = env.reset(seed=0)
    print(env.render())

    for step in range(10):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step + 1}")
        print(f"Action: {action}, Reward: {reward}")
        print(env.render())

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
