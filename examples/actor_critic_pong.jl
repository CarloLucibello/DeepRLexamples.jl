using Knet
using Gym
import AutoGrad: getval

const F = Float64

mutable struct History
    xsize
    nA::Int
    γ::F
    states::Array{F}
    actions::Vector{Int}
    rewards::Vector{F}
    Rlast
end

History(xsize, nA, γ) = History(xsize, nA, γ, zeros(F,0), zeros(Int, 0), zeros(F,0), 0)

function Base.push!(history, s, a, r)
    history.states = append!(history.states, s)
    push!(history.actions, a)
    push!(history.rewards, r)
end

function preprocess(I)
    I = I[36:195,:,:]
    I = I[1:2:end, 1:2:end, 1]
    I[I .== 144] .= 0
    I[I .== 109] .= 0
    I[I .!= 0] .= 1
    return vec(I)
end


function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return R
    return (R .- mean(R)) ./ (std(R) + F(1e-10)) #speeds up training a lot
end

function initweights(atype, xsize, ysize)
    @assert xsize == (80, 80, 1)
    w = []
    ch1 = 4
    ch2 = 8
    nh = 64
    push!(w, xavier(8, 8, xsize[3], ch1))
    push!(w, zeros(1, 1, ch1, 1))
    push!(w, xavier(4, 4, ch1, ch2))
    push!(w, zeros(1, 1, ch2, 1))
    
    push!(w, xavier(nh, 9*9*ch2))
    push!(w, zeros(nh, 1))

    push!(w, xavier(ysize, nh))
    push!(w, zeros(ysize, 1))
    push!(w, xavier(1, nh))
    push!(w, zeros(1, 1))
    
    return map(wi->convert(atype,wi), w)
end

function predict(w, x)
    x = reshape(x, 80, 80, 1, :)
    x = relu.(conv4(w[1], x, stride=4, padding=2) .+ w[2])
    x = relu.(conv4(w[3], x, stride=2) .+ w[4])
    x = mat(x)
    x = relu.(w[5]*x .+ w[6])
    prob_act = w[7] * x .+ w[8]
    value = w[9] * x .+ w[10]
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
    xsize, nA = history.xsize, history.nA
    nS = 80*80
    M = length(history.states)÷nS
    states = reshape(history.states, nS, M)
    R = discount(history.rewards, history.γ)
    
    p, V = predict(w, states)
    V = V'
    inds = history.actions + nA*(0:M-1)
    lp = logp(p, 1)[inds] # lp is a vector
    
    return -sum(lp .* (R .- getval(V))) / M + sum((R .- V).*(R .- V)) / M
end

function train!(w, history, opt)
    dw = grad(loss)(w, history)
    update!(w, dw, opt)
end

function main(;
    optim = Adam(lr=1e-2),
    γ = 0.99, #discount rate
    episodes = 500,
    rendered = true,
    seed = -1,
    infotime = 50)


    atype = gpu() >= 0 ? KnetArray{F} : Array{F}

    env = GymEnv("Pong-v0")
    seed > 0 && (srand(seed); srand(env, seed))
    xsize = (80, 80, 1)
    nA = 3 
    amap = Dict(1=>2, 2=>3, 3=>0) #up, down, no action

    w = initweights(atype, xsize, nA)
    opts = map(wi->deepcopy(optim), w)
    avgreward = -21.0
    for k=1:episodes
        state = reset!(env)
        episode_reward = 0
        history = History(xsize, nA, γ)
        prev_s = zeros(80*80)
        while true
            s = preprocess(state) .- prev_s
            p, V = predict(w, s)
            p = softmax(p)
            action = sample_action(p)

            next_state, reward, done, _ = step!(env, amap[action])
            push!(history, s, action, reward)
            state = next_state
            episode_reward += reward
            # @show done reward episode_reward
            k % infotime == 0 && rendered && render(env)
            done && break
        end
      
        avgreward = 0.1*episode_reward + 0.9*avgreward
        println("(episode:$k, Reward:$episode_reward, AvgReward:$(trunc(avgreward,3))")
        k % infotime == 0 && rendered && render(env, close=true)

        train!(w, history, opts)
    end
    return w
end
