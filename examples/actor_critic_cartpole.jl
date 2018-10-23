using Knet
import Gym
import AutoGrad
import Random
using Statistics

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
    push!(w, xavier(1, x)) # Value function
    push!(w, zeros(1, 1))
    
    return w
end

function predict(w, x)
    for i = 1:2:length(w)-4
        x = relu.(w[i] * x .+ w[i+1])
    end
    prob_act = w[end-3] * x .+ w[end-2]
    value = w[end-1] * x .+ w[end]
    return prob_act, value
end


function sample_action(probs)
    @assert size(probs, 2) == 1
    cprobs = cumsum(probs, dims=1)
    sampled = cprobs .> rand() 
    return mapslices(argmax, sampled, dims=1)[1]
end

function loss(w, history)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    R = discount(history.rewards, history.γ)
    
    p, V = predict(w, states)
    V = vec(V)
    A = R .- V   # advantage  
    inds = history.actions + nA*(0:M-1)
    lp = logsoftmax(p, dims=1)[inds] # lp is a vector

    return -mean(lp .* AutoGrad.value(A)) + L2Reg(A)
end

L2Reg(x) = mean(x .* x)

function main(;
    hidden = [100], # width inner layers
    lr = 1e-2,
    γ = 0.99, #discount rate
    episodes = 500,
    render = true,
    seed = -1,
    infotime = 50)

    env = Gym.GymEnv("CartPole-v1")
    seed > 0 && (Random.seed!(seed); Gym.seed!(env, seed))
    nS, nA = 4, 2 
    w = initweights(hidden, nS, nA)
    opt = [Adam(lr=lr) for _=1:length(w)]
    
    avgreward = 0
    for episode=1:episodes
        state = Gym.reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:10000
            p, V = predict(w, state)
            p = softmax(p, dims=1)
            action = sample_action(p)

            next_state, reward, done, info = Gym.step!(env, action-1)
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            state = next_state
            episode_rewards += reward

            episode % infotime == 0 && render && Gym.render(env)
            done && break
        end

        avgreward = 0.1 * episode_rewards + avgreward * 0.9
        if episode % infotime == 0
            println("(episode:$episode, avgreward:$avgreward)")
            Gym.close!(env)
        end
        
        dw = grad(loss)(w, history)
        update!(w, dw, opt)
    end

    return w
end
