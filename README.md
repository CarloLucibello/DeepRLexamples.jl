# DeepRLexamples
This repo provides examples of deep reinforcement learning in julia (v1.0 and above) using [Knet](https://github.com/denizyuret/Knet.jl) deep learning library and [OpenAI Gym](https://gym.openai.com/). Contributions are very welcome!

## Installation
Install the `gym` environment for python 
```bash
pip install --user gym[atari]
```
and the julia packages [Gym.jl](https://github.com/ozanarkancan/Gym.jl) and [Knet](https://github.com/denizyuret/Knet.jl)
```julia
] add Gym Knet
```

You are now ready to run any of the examples in the repo. You can clone the whole repo with

```julia
Pkg.clone("https://github.com/CarloLucibello/DeepRLexamples.jl")
```

## Usage
```julia
include("actor_critic_pong.jl")
main(seed=17, episodes=1000, lr=1e-2, render=true, infotime=50)
```

## Examples
- [reinforce_cartpole.jl](examples/reinforce_cartpole.jl): reinforce algorithm with a multi-layer perceptron. CPU only.

- [actor_critic_cartpole.jl](examples/actor_critic_cartpole.jl): actor critic algorithm with a multi-layer perceptron. CPU only.

- [actor_critic_pong.jl](examples/actor_critic_pong.jl): actor critic algorithm with a convolutional neural network. Following [Karphaty's blog entry](http://karpathy.github.io/2016/05/31/rl/), but using actor-critic instead of simple police gradient. Also, a convolutional neural network instead of a multi-layer perceptron. Runs on both CPU and GPU.