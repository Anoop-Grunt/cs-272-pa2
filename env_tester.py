import myenv

for seed in range(10):
    env = myenv.MyEnv()
    env.reset(seed=seed)

    observation, reward, terminated, truncated, info = env.step(2)

    print(
        f"seed={seed}, reward={reward}, "
        f"current={info['current_person']}, "
        f"infected={info['infected_count']}"
    )
