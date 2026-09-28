"""Task 1: your own custom Gymnasium environment.

Design the world yourself. The requirements it has to meet are in the assignment
readme.

Delete this docstring and describe your own world instead.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

import networkx as nx
from phart import ASCIIRenderer, LayoutOptions, NodeStyle


class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None):
        # TODO: describe your world here -- the map, the pieces, the constants.

        # TODO: set the two spaces. Both must be Discrete.

        self.observation_space = spaces.Discrete(384)
        self.action_space = spaces.Discrete(6)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        self.current_person = 0
        self.infected_mask = 1
        self.neighbors =     None
        self.infection_probability = None
   
    def _get_obs(self) -> int:

        return self.infected_mask * 6 + self.current_person
    
    def _get_info(self) -> dict:
        return {
            "current_person": self.current_person,
            "infected_count": self.infected_mask.bit_count(),
        }
    
    def _generate_graph(self) -> dict[int, set[int]]:
        neighbors = {
            person: set()
            for person in range(6)
        }
        def add_edge(first: int, second: int):
            neighbors[first].add(second)
            neighbors[second].add(first)
    
        # this is just a failsafe cuz the random seed  might actually disconnect components
        # so making sure a basic ring of edges always exists, by hardcoding
        for person in range(6):
            add_edge(person, (person + 1) % 6)
    
        for first in range(6):
            for second in range(first + 1, 6):
                if second in neighbors[first]:
                    continue
    
                if self.np_random.random() < 0.25:
                    add_edge(first, second)
    
        return neighbors

    def _generate_infection_probabilities(self) -> dict[int, float]:
        probabilities = self.np_random.uniform(
            low=0.40,
            high=0.90,
            size=6,
        )
    
        probabilities[0] = 1.0
    
        return {
            person: float(probabilities[person])
            for person in range(6)
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        # This line seeds self.np_random. Without it, seeding does not work and
        # the reproducibility test fails.
        super().reset(seed=seed)
        # TODO: put the world back to its starting state.
        if self.neighbors is None:
            self.neighbors = self._generate_graph()
            self.infection_probability = self._generate_infection_probabilities()
        else:
            self._generate_graph()
            self._generate_infection_probabilities()
    
        self.current_person = 0
        self.infected_mask = 1
    
        return self._get_obs(), self._get_info()
    



    def step(self, action: int):
        # TODO: apply the action, with noise drawn from self.np_random.
        #
        # Return terminated=True when the episode genuinely ends -- goal reached,
        # agent died, game over. Leave truncated as False and let the TimeLimit
        # wrapper from register() handle running out of time. The agent treats
        # the two differently, and so should you.

        action = int(action)
    
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
    
        target = action
        reward = -0.05
        terminated = False
    
        # The selected person is not directly connected.
        if target not in self.neighbors[self.current_person]:
            reward = -0.20
    
        # The target is already infected, so the virus can move there.
        elif self.infected_mask & (1 << target):
            self.current_person = target
    
        # The target is healthy, so attempt transmission.
        else:
            probability = self.infection_probability[target]
    
            if self.np_random.random() < probability:
                self.infected_mask |= 1 << target
                self.current_person = target
                reward = 1.0
    
                # End the episode after infecting five people.
                if self.infected_mask.bit_count() >= 5:
                    reward += 5.0
                    terminated = True
            else:
                reward = -1.0
                terminated = True
    
        truncated = False
    
        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            self._get_info(),
        )

    def render(self):
        """Return a readable picture of the current state, as a string."""
        if self.render_mode != "ansi":
            return None
        # TODO: draw it. You need this for the sample episode in your report.
        #
        if self.render_mode != "ansi":
            return None
    
        graph = nx.Graph()
        graph.add_nodes_from(range(6))
    
        # Add each undirected edge only once.
        for person, neighbors in self.neighbors.items():
            for neighbor in neighbors:
                if person < neighbor:
                    graph.add_edge(person, neighbor)
    
        def status(person: int) -> str:
            if person == self.current_person:
                return "V"
            if self.infected_mask & (1 << person):
                return "I"
            return "H"
    
        # Give each node its current display label.
        labels = {
            person: f"{person}:{status(person)}"
            for person in range(6)
        }
    
        labeled_graph = nx.relabel_nodes(graph, labels)
    
        diagram = ASCIIRenderer(labeled_graph).render()
    
        edges = sorted(
            tuple(sorted(edge))
            for edge in graph.edges()
        )
    
        connection_text = ", ".join(
            f"{first}-{second}"
            for first, second in edges
        )


        probability_text = ", ".join(
            f"{person}={probability:.2f}"
            for person, probability in sorted(self.infection_probability.items())
        )   
        return (
            "Virus spread network:\n"
            f"{diagram}\n\n"
            f"Current person: {self.current_person}\n"
            f"Infected count: {self.infected_mask.bit_count()}/6\n"
            f"Connections: {connection_text}\n"
            f"Infection probabilities: {probability_text}"
        )
    def close(self):
        pass


# TODO: name your environment. The id must start with "cs272/" and end with a
# version, and max_episode_steps must be large enough that a competent agent can
# finish but small enough that a lost one gives up.
register(
    id="cs272/MyEnv-v0",
    entry_point="myenv:MyEnv",
    max_episode_steps=300,
)
