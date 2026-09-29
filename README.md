# CS272 PA2: Gymnasium - Virus Spreading through a Social Network

## Important: Python Version

This project needs **Python 3.14**.

The reason is that the version of the `phart` package used for the ASCII graph renderer requires Python 3.14.

Install the packages with:

`pip install -r requirements.txt`

## Environment

The environment is a graph of six people.

Person `0` starts with the virus. At each step, the agent chooses a person to target.

If the target is connected to the current person, the virus tries to infect them. The infection can succeed or fail because each person has a different infection probability.

The goal is to infect everyone in the network.

The environment ID is:

`cs272/Virus-v0`

The full environment description is in [env.md](env.md).

## Agent

The agent uses tabular SARSA(λ).

It uses:

- Epsilon-greedy action selection
- Random tie-breaking
- Eligibility traces
- A Q-table
- Accumulating traces

The agent does not know anything specific about the virus environment. It only reads the number of states and actions from the Gymnasium spaces.

## Files

- `myenv.py` — The custom environment
- `myagent.py` — The SARSA(λ) agent
- `myrunner.py` — Training and experiments
- `env.md` — Full environment documentation
- `requirements.txt` — Python packages

## Running the Code

After activating the Python 3.14 virtual environment, run:

`python myrunner.py`

The runner:

1. Trains the agent with five different λ values.
2. Uses five random seeds for each λ.
3. Trains a random-agent baseline.
4. Creates a learning-curve plot.
5. Prints a results table.
6. Shows one greedy episode using the ANSI renderer.

## Experiment Settings

The main experiment uses:

| Setting | Value |
|---|---:|
| Environment | `cs272/Virus-v0` |
| Number of people | `6` |
| Episodes per run | `5000` |
| λ values | `0.0, 0.3, 0.6, 0.9, 1.0` |
| Seeds | `0, 1, 2, 3, 4` |
| Alpha | `0.05` |
| Gamma | `0.99` |
| Epsilon | `0.15` |
| Initial Q-value | `1.0` |
| Trace type | Accumulating |
| Moving-average window | `100` |

## Results

Running `myrunner.py` creates the learning plot:

`lambda_sweep.png`

The results table shows:

- When each λ first reaches the selected return target
- The average return near the end of training
- The average number of infected people near the end of training

The random agent is included as a baseline.

## Why Lambda Matters

When λ is close to zero, the agent mostly gives credit to the most recent decision.

When λ is larger, a reward can be given to earlier decisions in the same episode.

This matters because the agent may need to make several good moves before it successfully spreads the virus. However, larger traces can also give credit to bad decisions, especially because infection attempts are random.

## Reproducibility

The environment uses Gymnasium's seeding system.

The agent also has its own random-number generator.

The graph, infection probabilities, action choices, and infection results can be reproduced by using the same seeds.

