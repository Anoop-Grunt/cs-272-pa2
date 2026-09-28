import gymnasium as gym
import myenv


def main():
    env = gym.make(
        "cs272/MyEnv-v0",
        render_mode="ansi",
    )

    observation, info = env.reset(seed=2)
    print(env.render())

    for step_number in range(10):
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step_number + 1}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(env.render())

        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
