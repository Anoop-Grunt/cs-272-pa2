# Virus Spread Network Environment

## Overview

This environment models a virus spreading through a small social network.

Each node represents one person. Person `0` initially carries the virus, and the agent chooses which person to target next. The goal is to infect everyone in the network before making three failed infection attempts.

The environment is registered as:

`cs272/Virus-v0`

It can be created with:

`gym.make("cs272/Virus-v0")`

## Environment Arguments

The constructor is:

`Virus(render_mode=None, n_people=6)`

### `render_mode`

The supported render mode is:

`"ansi"`

When enabled, `env.render(render_mode="ansi")` returns a text-based representation of the network.

### `n_people`

The number of people in the network. The final assignment configuration uses six people.

The observation space has:

`2 ** n_people * n_people`

states. Therefore, values larger than six people exceed the assignment's observation-space limit of 500 states.

## Social Network

The environment uses an undirected graph.

A ring of connections is always created first so that every person belongs to one connected component. Additional edges are added randomly. Each possible non-ring edge is added with probability `0.25`.

The graph is generated when the environment is first reset and remains fixed for later episodes.

The infection probabilities are also generated once and remain fixed for the environment instance.

## Observation Space

For six people, the observation space is:

`spaces.Discrete(384)`

The observation is one integer encoding:

1. Which people are infected
2. The person currently carrying the virus (The assumption is that only this person, and not just anyone with the infection can actually infect other people)

The encoding is:

`observation = infected_mask * n_people + current_person`

The infected mask is a bitmask. Bit `i` is `1` when person `i` is infected and `0` otherwise.

To decode an observation:

`infected_mask = observation / n_people`

`current_person = observation % n_people`

For example, observation `203` decodes as:

- `infected_mask = 203 / 6 = 33`
- `current_person = 203 % 6 = 5`

The number `33` is binary `100001`, meaning people `0` and `5` are infected. The current virus carrier is person `5`.

The graph structure and infection probabilities are not included directly in the observation.

## Action Space

The action space is:

`spaces.Discrete(6)`

Each action represents a target person:

| Action | Target |
|---:|---|
| `0` | Person 0 |
| `1` | Person 1 |
| `2` | Person 2 |
| `3` | Person 3 |
| `4` | Person 4 |
| `5` | Person 5 |

The selected person must be connected to the current person for the action to be useful. Invalid actions are allowed but receive a negative reward.

## Starting State

At the beginning of every episode:

- Person `0` is infected.
- The current person is `0`.
- All other people are healthy.
- The failure counter is `0`.

The initial infected mask is `1`.

For six people, the initial observation is:

`1 * 6 + 0 = 6`

## Infection Probabilities

Person `0` has an infection probability of `1.0`.

The probabilities for the other people are sampled uniformly from:

`[0.10, 0.70)`

The probabilities are generated using the environment's random-number generator and remain fixed across episodes for the same environment instance.

## Transition Rules

At each step, the agent selects a target person.

### Invalid or Non-Neighbor Target

If the target is not directly connected to the current person:

- Reward: `-0.45`
- Current person: unchanged
- Infection mask: unchanged

### Already Infected Target

If the target is already infected:

- Reward: `-0.30`
- The virus moves to that person
- No new infection occurs

This penalty discourages repeatedly visiting already explored nodes.

### Healthy Target

If the target is healthy, the environment attempts transmission.

A random value is drawn from `[0, 1)`. Transmission succeeds when:

`random_value < target_probability`

#### Successful Transmission

If transmission succeeds:

- Reward: `+1.0`
- The target becomes infected
- The current person becomes the newly infected target

#### Failed Transmission

If transmission fails:

- Reward: `-0.75`
- The failure counter increases by one
- The current person does not change
- The infection mask does not change

The episode terminates after three failed infection attempts.

## Rewards

| Event | Reward |
|---|---:|
| Invalid or non-neighbor action | `-0.45` |
| Move to an already infected person | `-0.30` |
| Successful infection | `+1.00` |
| Failed infection attempt | `-0.75` |
| Infecting the final person | Additional `+3.00` |

The undiscounted return is the sum of all rewards received during an episode.

## Termination and Truncation

The episode terminates naturally when either of these conditions occurs:

1. Every person is infected.
2. The agent makes three failed infection attempts.

The environment itself always returns `truncated=False`.

The registered `TimeLimit` wrapper applies a maximum of 300 steps:

`max_episode_steps=300`

If the agent reaches this limit before a natural terminal condition, the wrapper returns `truncated=True`.

## Randomness and Seeding

All environment randomness uses `self.np_random`.

Randomness is used for:

- Generating additional graph connections
- Generating infection probabilities
- Determining whether an infection attempt succeeds

The environment does not use Python's `random` module or bare `numpy.random`.

Resetting with a seed makes behavior reproducible:

`env.reset(seed=0)`

Using the same seed and the same sequence of actions produces the same trajectory.

## Rendering

The environment supports ANSI rendering:

`env = gym.make("cs272/Virus-v0", render_mode="ansi")`

Calling `env.render(render_mode="ansi")` returns a text representation of the social network.

The rendering shows:

- The graph connections
- Each person's current status
- The current virus carrier
- The number of infected people
- The infection probabilities

The node labels use:

- `VIRUS`: current virus carrier
- `INFECTED`: infected
- `HEALTHY`: healthy

Rendering is used for debugging and for showing a sample greedy episode in the report. It is not used by the learning agent.

The rendering mode `human` is supported as well, but is mainly for reporting purposes.
