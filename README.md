# DeepRLexamples
This repo provides examples of deep reinforcement learning in julia using [Knet](https://github.com/denizyuret/Knet.jl) deep learning library and [OpenAI Gym](https://gym.openai.com/). 

## Installation
Install the [Gym](https://github.com/CarloLucibello/Gym.jl) environment
```julia
Pkg.clone("https://github.com/CarloLucibello/Gym.jl")
```
and run any of the examples or clone the whole repo with

```julia
Pkg.clone("https://github.com/CarloLucibello/DeepRLexamples.jl")
```

## Usage
```julia
include("actor_critic_pong.jl"); using Knet

main(seed=1, episodes=1000, opt=Adam(lr=1e-2))
```

## Examples
- reinforce_cartpole: reinforce algorithm with a multi layer perceptron.

- actor_critic_cartpole: actor critic algorithm with a multi layer perceptron.

- actor_critic_pong: actor critic algorithm with a convolutional neural network. 


## Interface
This repo is just a collection of examples and
can be adapted to any use. Nonetheless,
to add new examples to the repo, and as a general
good practice, we reccomend any user-defined new enviroment 
to provide the following interface

```julia
"Reset an environment and returns an initial state (or observation)"
reset!(env) --> obs

"""
Take an environment, and an action to execute, and
step the environment forward. Return an observation, reward,
terminal flag and info.  
Depending on the environment, the observation could be the whole current state.
"""
step!(env, a) --> obs, rew, isdone, info

"Return an action space object that can be sampled with rand."
action_space(env) --> A

"Seeds the enviroment."
srand(env, seed)
```
and  optionally also
```julia
"Return true if the current state is terminal"
done(env) --> isdone

"The current state of the environment"
env.state

"Renders a graphic of the environment"
render(env)
```

The environment supplied by [Gym](https://github.com/CarloLucibello/Gym.jl) implements the above methods and some others.
