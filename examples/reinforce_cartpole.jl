using Knet
using Gym
import AutoGrad: getval

mutable struct History
    nS::Int
    nA::Int
    γ::Float64
    states::Vector{Float64}
    actions::Vector{Int}
    rewards::Vector{Float64}
end
History(nS, nA, γ) = History(nS, nA, γ, zeros(0),zeros(Int, 0),zeros(0))

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return R
    return (R .- mean(R)) ./ (std(R) + 1e-10) #speeds up training a lot
end

function initweights(hidden, xsize=4, ysize=2)
    w = []
    x = xsize
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, zeros(y, 1))
        x = y
    end
    push!(w, xavier(ysize, x)) # Prob actions
    push!(w, zeros(ysize, 1))
    
    return w
end

function predict(w, x)
    for i = 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
    end
    probs = w[end-1] * x .+ w[end]
    return probs
end

function softmax(x)
    y = maximum(x, 1)
    y = x .- y
    y = exp.(y)
    return y ./ sum(y, 1)
end

function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, 1)
    sampled = cprobs .> rand() 
    return mapslices(indmax, sampled, 1)[1]
end

function loss(w, history)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    R = discount(history.rewards, history.γ)
    
    p = predict(w, states)
    inds = history.actions + nA*(0:M-1)
    lp = logp(p, 1)[inds] # lp is a vector

    return -mean(lp .* R)
end

function main(;
    hidden = [100], # width inner layers
    lr = 1e-2, # learning rate
    γ = 0.99, #discount rate
    episodes = 500,
    rendered = true,
    seed = -1,
    infotime = 50)

    env = GymEnv("CartPole-v1")
    seed > 0 && (srand(seed); srand(env, seed))
    nS, nA = 4, 2 
    w = initweights(hidden, nS, nA)
    opt = [Adam(lr=lr) for _=1:length(w)]
    
    avgreward = 0
    for episode=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            p = predict(w, state)
            p = softmax(p)
            action = sample_action(p)

            next_state, reward, done, _ = step!(env, action_space(env)[action])
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            state = next_state
            episode_rewards += reward

            episode % infotime == 0 && rendered && render(env)
            done && break
        end

        avgreward = 0.02 * episode_rewards + avgreward * 0.98
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            rendered && render(env, close=true)
        end

        dw = grad(loss)(w, history)
        update!(w, dw, opt)
    end

    return w
end
