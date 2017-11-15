using Knet
using Gym
import AutoGrad: getval

const F = Float64

mutable struct History
    nS::Int
    nA::Int
    γ::F
    states::Vector{F}
    actions::Vector{Int}
    rewards::Vector{F}
end
History(nS, nA, γ) = History(nS, nA, γ, zeros(F,0),zeros(Int, 0),zeros(F,0))

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return R
    return (R .- mean(R)) ./ (std(R) + F(1e-10)) #speeds up training a lot
end

function initweights(atype, hidden, xsize=4, ysize=2)
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
    
    return map(wi->convert(atype,wi), w)
end

function predict(w, x)
    for i = 1:2:length(w)-4
        x = relu.(w[i] * x .+ w[i+1])
    end
    prob_act = w[end-3] * x .+ w[end-2]
    value = w[end-1] * x .+ w[end]
    return prob_act, value
end

function softmax(x)
    y = maximum(x, 1)
    y = x .- y
    y = exp.(y)
    return y ./ sum(y, 1)
end

function sample_action(probs)
    probs = Array(probs)
    cprobs = cumsum(probs)
    sampled = cprobs .> rand() 
    actions = mapslices(indmax, sampled, 1)
    return actions[1]
end

function loss(w, history)
    nS, nA = history.nS, history.nA
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    R = discount(history.rewards, history.γ)
    
    p, V = predict(w, states)
    V = vec(V)
    inds = history.actions + nA*(0:M-1)
    lp = logp(p, 1)[inds] # lp is a vector
    
    return -sum(lp .* (R .- getval(V))) / M + sum((R .- V).*(R .- V)) / M
end

function train!(w, history, opt)
    dw = grad(loss)(w, history)
    update!(w, dw, opt)
end

function main(;
    hidden = [100],
    optim = Adam(lr=1e-2),
    γ = 0.99, #discount rate
    episodes = 500,
    rendered = true,
    seed = -1,
    infotime = 50)


    atype = gpu() >= 0 ? KnetArray{F} : Array{F}

    env = GymEnv("CartPole-v1")
    seed > 0 && (srand(seed); srand(env, seed))
    nS, nA = 4, 2 
    w = initweights(atype, hidden, nS, nA)
    opts = map(wi->deepcopy(optim), w)

    avgreward = 0
    for k=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(nS, nA, γ)
        for t=1:1000
            s = convert(atype, state)
            p, V = predict(w, s)
            probs = softmax(p)
            action = sample_action(probs)

            next_state, reward, done, _ = step!(env, action_space(env)[action])
            append!(history.states, state)
            push!(history.actions, action)
            push!(history.rewards, reward)
            state = next_state
            episode_rewards += reward

            k % infotime == 0 && rendered && render(env)
            done && break
        end

        avgreward = 0.02 * episode_rewards + avgreward * 0.98

        k % infotime == 0 && println("(episode:$k, avgreward:$avgreward)")
        k % infotime == 0 && rendered && render(env, close=true)

        train!(w, history, opts)
    end
    return w
end
