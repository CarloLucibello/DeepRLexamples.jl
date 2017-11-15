using Knet
using Gym
import AutoGrad: getval

# using TimerOutputs
# reset_timer!()

const F = Float32

mutable struct History
    xsize
    nA::Int
    γ::F
    states  # flattens all states into a vector
    actions::Vector{Int}
    rewards::Vector{F}
end

History(xsize, nA, γ, atype) = History(xsize, nA, γ, 
                                convert(atype, zeros(F,0)), 
                                zeros(Int, 0), 
                                zeros(F,0))

function Base.push!(history, s, a, r)
    history.states = [history.states; s]
    push!(history.actions, a)
    push!(history.rewards, r)
end

function preprocess(I, atype)
    I = I[36:195,:,:]
    I = I[1:2:end, 1:2:end, 1]
    I[I .== 144] .= 0
    I[I .== 109] .= 0
    I[I .!= 0] .= 1
    return convert(atype, vec(I))
end

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return R
    # return R ./ 21
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

    push!(w, xavier(ysize, nh)) # policy
    push!(w, zeros(ysize, 1))
    push!(w, xavier(1, nh)) # value
    push!(w, zeros(1, 1))
    
    return map(wi->convert(atype,wi), w)
end

function predict(w, x)
    x = reshape(x, 80, 80, 1, length(x)÷(80*80))
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
    nS = prod(xsize)
    M = length(history.states) ÷ nS
    states = reshape(history.states, nS, M)

    p, V = predict(w, states)
    V = vec(V)   

    R = discount(history.rewards, history.γ)
    R = convert(typeof(getval(V)), R)
    A = R .- V   # advantage  

    inds = history.actions + nA*(0:M-1)
    lp = logp(p, 1)[inds] # lp is a vector

    return -mean(lp .* getval(A)) + L2Reg(A)
end

L2Reg(x) = mean(x .* x)

function main(;
        optim = Adam(lr=1e-2),
        γ = 0.99, #discount rate
        episodes = 10000,  # max episodes played
        rendered = false, # if true display an episode every infotime ones
        seed = -1,
        infotime = 50,
        atype = gpu() >= 0 ? KnetArray{F} : Array{F},
    )

    info("using $atype")
    env = GymEnv("Pong-v0")
    seed > 0 && (srand(seed); srand(env, seed))
    xsize = (80, 80, 1) # input dimensions of the policy network
    nA = 3 # number of actions. Could also set to 2
    amap = Dict(1=>2, 2=>3, 3=>0) #up, down, no action

    w = initweights(atype, xsize, nA)
    opt = map(wi->deepcopy(optim), w)
    avgreward = -21.0
    for episode=1:episodes
        state = reset!(env)
        episode_reward = 0
        history = History(xsize, nA, γ, atype)
        prev_s = convert(atype, zeros(prod(xsize)))
        while true
            curr_s = preprocess(state, atype)
            s = curr_s .- prev_s
            prev_s = curr_s

            p, V = predict(w, s)
            p = softmax(p)
            action = sample_action(p)
            
            next_state, reward, done, _ = step!(env, amap[action])           
            
            push!(history, s, action, reward)
            state = next_state
            episode_reward += reward
            episode % infotime == 0 && rendered && render(env)
            done && break
        end
      
        avgreward = 0.02*episode_reward + 0.98*avgreward
        println("Episode:$episode, Reward:$episode_reward, AvgReward:$(trunc(avgreward,3))")
        episode % infotime == 0 && rendered && render(env, close=true)

        dw = grad(loss)(w, history)
        update!(w, dw, opt)    
    end
    
    return w
end
