import matplotlib.pyplot as plt
import networkx as nx
import gymnasium as gym
import myenv


def draw_environment(env, save_path=None):
    """Draw the current virus-spread graph."""

    base_env = env.unwrapped

    graph = nx.Graph()
    graph.add_nodes_from(range(base_env.n_people))

    for person, neighbors in base_env.neighbors.items():
        for neighbor in neighbors:
            if person < neighbor:
                graph.add_edge(person, neighbor)

    positions = nx.kamada_kawai_layout(graph)

    node_colors = []
    labels = {}

    for person in range(base_env.n_people):
        if person == base_env.current_person:
            node_colors.append("gold")
            status = "VIRUS"
        elif base_env.infected_mask & (1 << person):
            node_colors.append("red")
            status = "INFECTED"
        else:
            node_colors.append("skyblue")
            status = "HEALTHY"

        probability = base_env.infection_probability[person]

        labels[person] = (
            f"{person}\n"
            f"{status}\n"
            f"p={probability:.2f}"
        )

    plt.figure(figsize=(10, 7))

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color="gray",
        width=2,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=2600,
        edgecolors="black",
        linewidths=1.5,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels=labels,
        font_size=8,
        font_weight="bold",
    )

    infected_count = base_env.infected_mask.bit_count()

    plt.title(
        f"Virus Spread Network "
        f"({infected_count}/{base_env.n_people} infected)"
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def main():
    env = gym.make("cs272/MyEnv-v0")

    env.reset(seed=0)

    print("Initial graph")
    draw_environment(
        env,
        save_path="virus_network_step_0.png",
    )

    actions = [1, 2, 3, 4, 5]

    for step_number, action in enumerate(actions, start=1):
        observation, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {step_number}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Infected count: {info['infected_count']}")

        draw_environment(
            env,
            save_path=f"virus_network_step_{step_number}.png",
        )

        if terminated or truncated:
            print("Episode ended.")
            break

    env.close()
if __name__ == "__main__":
    main()
