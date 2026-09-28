import myenv


def canonical_graph(neighbors):
    return {
        person: tuple(sorted(connected_people))
        for person, connected_people in sorted(neighbors.items())
    }


# Same seed in fresh environments should produce the same graph.
env_a = myenv.MyEnv(render_mode="ansi")
env_b = myenv.MyEnv(render_mode="ansi")

env_a.reset(seed=42)
env_b.reset(seed=42)

graph_a = canonical_graph(env_a.neighbors)
graph_b = canonical_graph(env_b.neighbors)

print("Same seed, same graph:", graph_a == graph_b)
print(graph_a)

# Resetting the same environment should preserve its graph.
original_graph = canonical_graph(env_a.neighbors)

env_a.reset(seed=999)

new_graph = canonical_graph(env_a.neighbors)

print("Graph preserved across reset:", original_graph == new_graph)

# Print the rendered graph.
print("\nRendered graph:")
print(env_a.render())

env_a.close()
env_b.close()
