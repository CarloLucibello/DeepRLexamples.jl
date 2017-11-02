__precompile__()
module DeepRL

using POMDPs

export AbstractEnvironment

# mandatory interface
export reset!, step!, actions, state #,done

# optional overloads
export sample_action, n_actions, render, finished

# POMDPs integration
export POMDPEnvironment, MDPEnvironment, obs_dimensions

abstract type AbstractEnvironment end

"""
    reset!(env)

Reset an environment and returns an initial state (or observation).
"""
reset!(env::AbstractEnvironment) = error("not implemented for this environment!")


"""
    step!(env, a) --> obs, rew, done, info

Take an environment, and an action to execute, and
step the environment forward. Return an observation, reward,
terminal flag and info.

Depending on the environment, the observation could be the whole current state.
"""
step!(env::AbstractEnvironment, a) = error("not implemented for this environment!")

"""
    actions(env)

Return an action space object that can be sampled with rand.
"""
actions(env::AbstractEnvironment) = error("not implemented for this environment!")

############### OPTIONAL OVERLOADS #########################################

"""
    finished(env)

Return true if the current state is terminal.
"""
finished(env::AbstractEnvironment) = error("not implemented for this environment!")

"""
    state(env)

Return the current state of the environment.
"""
state(env::AbstractEnvironment) = env.state

"""
    sample_action(env)

Sample an action from the action space of the environment.
"""
sample_action(env::AbstractEnvironment) = rand(actions(env))

"""
    render(env)

Renders a graphic of the environment.
"""
render(env::AbstractEnvironment) = error("not implemented for this environment!")
########################################################

include("pomdps_integration.jl")
# include("ZMQServer.jl")


end # module
