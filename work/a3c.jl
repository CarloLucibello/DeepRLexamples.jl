using Knet
using Gym
using DeepRL
import AutoGrad: getval
import ImageTransformations: imresize
import PyPlot: imshow

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
History(xsize, nA, γ) = History(xsize, nA, γ, zeros(F,xsize..., 0), zeros(Int, 0), zeros(F,0), 0)

function Base.push!(history, s, a, r)
    history.states = cat(4, history.states, s)
    push!(history.actions, a)
    push!(history.rewards, r)
end

mutable struct Preprocess
    lastimg  # last state
    x        # 84x84xTx1 tensor of rescaled and stacked last T frames

    Preprocess(imgsize, xsize) = new(zeros(F, imgsize), zeros(F, xsize..., 1))
end

"Prepocessing according to 
Human-level control through deep reinforcement learning"
function (pr::Preprocess)(img; reset=false)
    img = convert(Array{F}, img)
    x = max.(img, pr.lastimg) # flickering smoothing
    kernel = reshape(F[0.299,0.587,0.114], 1, 1, 3, 1) # RGB -> Y′
    
    x = conv4(kernel, reshape(x, size(x)..., 1))
    x = reshape(x, size(img)[1:2]...)  # imresize gives NaN otherwise
    x = imresize(x, (84, 84))  
    if reset
        sz = size(pr.x)
        x = cat(3, [x for _=1:sz[3]]...)
        pr.x = reshape(x, sz)
    else
        pr.x = cat(3, x, pr.x[:,:,1:end-1,:])
    end
    @assert size(pr.x) == (84, 84, 4,1)
    
    pr.lastimg = img  
    return pr.x
end

function discount(rewards, Rlast, γ)
    # @show rewards Rlast
    R = similar(rewards)
    R[end] = rewards[end] + γ * Rlast
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    # return (R .- mean(R)) ./ (std(R) + F(1e-8))
    return R
end

function entropy(p)
    @assert all(isfinite, p)
    s = @. p*log(p+F(1e-10)) + (1-p)*log(1-p+F(1e-10))
    # s = @. p*log(p) + (1-p)*log(1-p)
    return -sum(s) / length(p)
end

function initweights(atype, xsize, ysize)
    @assert xsize == (84, 84, 4)
    w = []
    ch1 = 8
    ch2 = 16
    nh = 128
    push!(w, xavier(8, 8, 4, ch1))
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
    x = relu.(conv4(w[1], x, stride=4) .+ w[2])
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
    sampled = cprobs .> rand() #TODO 
    actions = mapslices(indmax, sampled, 1)
    return actions[1]
end

function loss(w, history)
    R = discount(history.rewards, history.Rlast, history.γ)
    R = R'
    M = size(history.states, 4)
    states = history.states
    
    p, V = predict(w, states)
    p = softmax(p)
    inds = history.actions + history.nA*(0:M-1)
    # lp = logp(p, 1)[inds]
    lp = log.(p)[inds]
    lp = -sum(lp .* (R .- getval(V))) / M
    mse = sum((R .- V).*(R .- V)) / M

    # rand() < 0.1 && (@show getval(entropy(p)))
    # rand() < 0.1 && (@show getval(lp))
    # rand() < 0.1 && (@show getval(mse))
    # return -1e-3 * entropy(p) 
    return lp + mse  - 1e-3*entropy(p) 
    # return -1e-3*entropy(p)
end

function train!(w, history, opt)
    dw = grad(loss)(w, history)
    update!(w, dw, opt)
end

function main(;
    optim = Adam(lr=1e-3),
    γ = 0.99, #discount rate
    Tmax = Inf,
    tmax = 5,
    rendered = true,
    seed = -1,
    infotime = 50)


    atype = gpu() >= 0 ? KnetArray{F} : Array{F}

    env = GymEnv("Pong-v0")
    seed > 0 && (srand(seed); srand(env, seed))
    imgsize = size(reset!(env))
    xsize = (84, 84, 4)
    nA = 3 
    amap = Dict(1=>2, 2=>3, 3=>0) #up, down, no action
    
    w = initweights(atype, xsize, nA)
    opt = map(wi->deepcopy(optim), w)
    
    preprocess = Preprocess(imgsize, xsize)
    state = preprocess(reset!(env), reset=true)
    done = false
    T, t = 1, 1
    episode_reward = 0
    episode = 1
    while true
        if done
            println("(episode:$episode, Reward:$episode_reward)")
            if episode % infotime == 0
                rendered && render(env, close=true)
            end    
            episode_reward = 0
            episode += 1
            preprocess = Preprocess(imgsize, xsize)
            state = preprocess(reset!(env), reset=true)
        end
        history = History(xsize, nA, γ)
        tstart = t
        while true
            # if (t-1) % 4 != 0 #frame skip TODO
            if (t+1 - tstart) % 4 == 0
                action = history.actions[end]
            else
                p, V = predict(w, state)
                probs = softmax(p)
                action = sample_action(probs)
            end
                # rand() < 0.3 && (action = rand(1:3))
            next_state, reward, done, _ = step!(env, amap[action])
            push!(history, state, action, reward)
            state = preprocess(next_state)
            episode_reward += reward
            t += 1; T += 1
            episode % infotime == 0 && rendered && render(env)
            (done || (t - tstart == tmax)) && break
        end
        history.Rlast = done ? 0 : predict(w, state)[2][1] # bootstrap future times
        train!(w, history, opt)
        
        T > Tmax && break
    end
    return w
end
