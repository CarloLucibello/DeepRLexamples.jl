# DeepRL

This package provides a lean interface for reinforcement problems modelled afeter that of [OpenAI Gym](https://gym.openai.com/). 
This allows algorithms that work with Gym to be used with problems that are defined in this interface and vice versa.
In the `examples/` folder are some neat examples of deep reinforcement learning using the `Knet` julia library.

# Installation
```julia
Pkg.clone("https://github.com/CarloLucibello/DeepRL.jl")
Pkg.clone("https://github.com/CarloLucibello/Gym.jl") #for an OpenAIGym wrapper compliant to this interface. 
```

## Interface
New enviroments should inherit from `AbstractEnviroment` and implement the methods defined in `src/DeepRL.jl`.

Example:

```julia
using Gym 
using DeepRL

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
