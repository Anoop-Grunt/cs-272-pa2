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
import re

class MyEnv(gym.Env):
    """TODO: one line on what this world is and what the agent is trying to do."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, n_people: int = 6):
        # TODO: describe your world here -- the map, the pieces, the constants.
        # TODO: set the two spaces. Both must be Discrete.
        if n_people < 3:
            raise ValueError("n_people must be at least 3")
        self.n_people = n_people
        self.goal_count = n_people
        self.observation_space = spaces.Discrete(
            (2 ** n_people) * n_people
        )
        self.max_failures = 3
        self.failure_count = 0
        self.action_space = spaces.Discrete(n_people)
   
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode}")
        self.render_mode = render_mode

        self.current_person = 0
        self.infected_mask = 1
        self.neighbors =     None
        self.infection_probability = None
   
    def _get_obs(self) -> int:
        return self.infected_mask * self.n_people + self.current_person
    
    def _get_info(self) -> dict:
        return {
            "current_person": self.current_person,
            "infected_count": self.infected_mask.bit_count(),
        }
    
    def _generate_graph(self) -> dict[int, set[int]]:
        neighbors = {
            person: set()
            for person in range(self.n_people)
        }
        def add_edge(first: int, second: int):
            neighbors[first].add(second)
            neighbors[second].add(first)
    
        # this is just a failsafe cuz the random seed  might actually disconnect components
        # so making sure a basic ring of edges always exists, by hardcoding
        for person in range(self.n_people):
            add_edge(person, (person + 1) % self.n_people)
    
        for first in range(self.n_people):
            for second in range(first + 1, self.n_people):
                if second in neighbors[first]:
                    continue
                if self.np_random.random() < 0.25:
                    add_edge(first, second)
    
        return neighbors

    def _generate_infection_probabilities(self) -> dict[int, float]:
        probabilities = self.np_random.uniform(
            low=0.10,
            high=0.90,
            size=self.n_people,
        )
        probabilities[0] = 1.0
        return {
            person: float(probabilities[person])
            for person in range(self.n_people)
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
        self.failure_count = 0
    
        return self._get_obs(), self._get_info()
    


    def step(self, action: int):
        action = int(action)
    
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
    
        target = action
        reward = -0.20
        terminated = False
    
        #Invalid/non-neighbor action
        if target not in self.neighbors[self.current_person]:
            reward = -0.45
    
        #Moving to an already infected person
        elif self.infected_mask & (1 << target):
            self.current_person = target
            reward = -0.30
    
        #Healthy target, so we can attempt infection
        else:
            probability = self.infection_probability[target]
    
            #positive rewards if we actually succeed in infecting
            if self.np_random.random() < probability:
                self.infected_mask |= 1 << target
                self.current_person = target
                reward = 1.0
    
                if self.infected_mask.bit_count() >= self.goal_count:
                    reward += 3.0
                    terminated = True
    
            else:
                reward = -0.75
                self.failure_count += 1
    
                if self.failure_count >= self.max_failures:
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
        graph.add_nodes_from(range(self.n_people))
    
        # Add each undirected edge only once.
        for person, neighbors in self.neighbors.items():
            for neighbor in neighbors:
                if person < neighbor:
                    graph.add_edge(person, neighbor)
    
        def status(person: int) -> str:
            if person == self.current_person:
                return "VIRUS"
            if self.infected_mask & (1 << person):
                return "INFECTED"
            return "HEALTHY"
    
        # Give each node its current display label.
        labels = {
            person: f"{person}:{status(person)}"
            for person in range(self.n_people)
        }
    
        labeled_graph = nx.relabel_nodes(graph, labels)
    
        options = LayoutOptions(
            layout_strategy="kamada-kawai",
        )
        
        renderer = ASCIIRenderer(
            labeled_graph,
            options=options,
        )
        
        diagram = renderer.render()    

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
            f"Infected count: {self.infected_mask.bit_count()}/{self.n_people}\n"
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
