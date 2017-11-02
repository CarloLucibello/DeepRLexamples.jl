
mutable struct MDPEnvironment{S} <: AbstractEnvironment
    problem::MDP
    state::S
    rng::AbstractRNG
end
function MDPEnvironment(problem::MDP; rng::AbstractRNG=MersenneTwister(0))
    return MDPEnvironment(problem, initial_state(problem, rng), rng)
end

mutable struct POMDPEnvironment{S} <: AbstractEnvironment
    problem::POMDP
    state::S
    rng::AbstractRNG
end
function POMDPEnvironment(problem::POMDP; rng::AbstractRNG=MersenneTwister(0))
    return POMDPEnvironment(problem, initial_state(problem, rng), rng)
end

"""
    reset!(env::MDPEnvironment)

Reset an MDP environment by sampling an initial state and returning it.
"""
function reset!(env::MDPEnvironment)
    s = initial_state(env.problem, env.rng)
    env.state = s
    return convert_s(Array{Float64, 1}, s, env.problem)
end

"""
    reset!(env::POMDPEnvironment)

Reset an POMDP environment by sampling an initial state,
generating an observation and returning it.
"""
function reset!(env::POMDPEnvironment)
    s = initial_state(env.problem, env.rng)
    env.state = s
    o = generate_o(env.problem, s, env.rng)
    return convert_o(Array{Float64, 1}, o, env.problem)
end

function step!(env::MDPEnvironment, a)
    s, r = generate_sr(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert_s(Array{Float64, 1}, s, env.problem)
    return obs, r, t, info
end

function step!(env::POMDPEnvironment, a)
    s, o, r = generate_sor(env.problem, env.state, a, env.rng)
    env.state = s
    t = isterminal(env.problem, s)
    info = nothing
    obs = convert_o(Array{Float64, 1}, o, env.problem)
    return obs, r, t, info
end

function actions(env::Union{POMDPEnvironment, MDPEnvironment})
    return POMDPs.actions(env.problem)
end

"""
    sample_action(env::Union{POMDPEnvironment, MDPEnvironment})
Sample an action from the action space of the environment.
"""
function sample_action(env::Union{POMDPEnvironment, MDPEnvironment})
    return rand(env.rng, actions(env))
end

"""
    n_actions(env::Union{POMDPEnvironment, MDPEnvironment})
Return the number of actions in the environment (environments with discrete action spaces only)
"""
function POMDPs.n_actions(env::Union{POMDPEnvironment, MDPEnvironment})
    return n_actions(env.problem)
end


function obs_dimensions(env::MDPEnvironment)
    return size(convert_s(Array{Float64,1}, initial_state(env.problem, env.rng), env.problem))
end


function obs_dimensions(env::POMDPEnvironment)
    return size(convert_o(Array{Float64,1}, generate_o(env.problem, initial_state(env.problem, env.rng), env.rng), env.problem))
end
