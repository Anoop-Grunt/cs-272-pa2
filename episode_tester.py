import myenv


def run_episode(seed):
    env = myenv.Virus()
    observation, info = env.reset(seed=seed)

    trajectory = []

    for action in [2, 3]:
        observation, reward, terminated, truncated, info = env.step(action)

        trajectory.append(
            (observation, reward, terminated, truncated, info.copy())
        )

        if terminated or truncated:
            break

    env.close()
    return trajectory


first = run_episode(123)
second = run_episode(123)

print(first)
print(second)
print("Same trajectory:", first == second)



# import gymnasium as gym
# import myenv
#
# env = gym.make("cs272/MyEnv-v0")
# observation, info = env.reset(seed=0)
#
# for step_number in range(300):
#     # Action 0 is invalid from person 0 but does not terminate the episode.
#     observation, reward, terminated, truncated, info = env.step(0)
#
#     if terminated or truncated:
#         print("Ended after:", step_number + 1, "steps")
#         print("terminated:", terminated)
#         print("truncated:", truncated)
#         break
#
# env.close()
