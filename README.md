# DeepRL
This repo provides examples of deep reinforcement learning in julia using [Knet](https://github.com/denizyuret/Knet.jl) deep learning lybrary and [OpenAI Gym](https://gym.openai.com/). 

# Installation
Install the Gym enviroment
```julia
Pkg.clone("https://github.com/CarloLucibello/Gym.jl")
```
and run any of the examples or clone the whole repo with

```
Pkg.clone("https://github.com/CarloLucibello/DeepRL.jl")
```

## Interface
This repo is just a collection of examples and
can be adapted to any use. Nonetheless,
to add new examples to the repo, and as a general
good practice, we reccoment any user defined new enviroment 
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
step!(env, a) --> obs, rew, done, info

"Return an action space object that can be sampled with rand."
actions(env) --> A

"Seeds the enviroment."
srand(env, seed)
```
and  optionally also
```julia
"Return true if the current state is terminal"
finished(env) --> done

"The current state of the environment"
env.state

"Renders a graphic of the environment"
render(env)
```

The enviroment supplied by `Gym` implements all the above methods and some others.

```julia
using Gym 

env = GymEnv("CartPole-v0")

nsteps = 1
episodes = 10

for episode=1:episodes
    o = reset!(env)
    r_tot = 0.0
    for step=1:nsteps
        action = rand(actions(env))
        obs, rew, done, info = step!(env, action)
        println(obs, " ", rew, " ", done, " ", info)
        r_tot += rew
        done && break
    end
end
```
