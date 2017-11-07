using Knet
using Gym
using DeepRL
import AutoGrad: getval

const F = Float64

mutable struct SKEnv
    σ
    J
    E
    rng
end

function SKEnv(N, rng=Base.Random.GLOBAL_RNG)
    σ = ones(N)
    J = zeros(F, N, N)
    for j=1:N, i=j+1:N
        J[i, j] = J[j, i] = rand(rng,[-1,1]) / √N
    end
    return SKEnv(σ, J, 0, rng) 
end

function reset!(env::SKEnv)
    rand!(env.rng, env.σ, [-1,1])
    env.E = energy(env.J, env.σ)
    return env.σ
end

Base.srand(env::SKEnv, seed) = srand(env.rng, seed)

function step!(env::SKEnv, a) 
    # the action corresponds to the new state itself
    env.σ = a; 
    E = energy(env.J, a)
    ΔE = E - env.E
    env.E = E
    a, -ΔE, false, nothing
end

energy(J, σ) = -(σ' * J * σ)[1] / (2size(J,1))

mutable struct History
    N::Int
    γ::F
    states::Vector{F}
    actions::Vector{Int}
    rewards::Vector{F}
end
History(N, γ) = History(N, γ, zeros(Int,0), zeros(Int, 0), zeros(F,0))

function discount(rewards, γ)
    R = similar(rewards)
    R[end] = rewards[end]
    for k = length(rewards)-1:-1:1
        R[k] = γ * R[k+1] + rewards[k]
    end
    return (R .- mean(R)) ./ (std(R) + F(1e-8))
end

function initweights(atype, hidden, N)
    w = []
    x = N
    for y in [hidden...]
        push!(w, xavier(y, x))
        push!(w, zeros(y, 1))
        x = y
    end
    push!(w, xavier(N, x)) # Prob actions
    push!(w, zeros(N, 1))
    push!(w, xavier(1, x)) # Value function
    push!(w, zeros(1, 1))
    
    return map(wi->convert(atype,wi), w)
end

function predict(w, x)
    for i = 1:2:length(w)-4
        x = relu.(w[i] * x .+ w[i+1])
    end
    prob_act = sigm.(w[end-3] * x .+ w[end-2])
    value = w[end-1] * x .+ w[end]
    return prob_act, value
end

function sample_action(probs)
    # the action corresponds to the new state itself
    N = length(probs)
    a = Vector{Int}(N)
    r = rand(N)
    for i=1:N
        a[i] = r[i] < probs[i] ? 1 : -1
    end
    return a
end


function loss(w, history)
    N = history.N
    M = length(history.states)÷N
    s = reshape(history.states, N, M)
    a = (1 .+ reshape(history.actions, N, M)) ./ 2
    R = discount(history.rewards, history.γ)
    R = R'
    p, V = predict(w, s)
    BCE = @. a * log(p + F(1e-10)) + (1-a) * log(1 - p + F(1e-10)) 
    BCE = -sum(BCE .* (R .- getval(V))) / (N*M)
    return BCE + sum((R .- V) .* (R .- V)) / M
end

function train!(w, history, opt)
    dw = grad(loss)(w, history)
    update!(w, dw, opt)
end

function main(;
    N=10,
    tmax=100 , #episode length
    hidden = [100],
    optim = Adam(lr=1e-2),
    γ = 0.99, #discount rate
    episodes = 500,
    seed = -1,
    infotime = 1)


    atype = gpu() >= 0 ? KnetArray{F} : Array{F}
    seed > 0 && srand(seed)
    env = SKEnv(N)
    seed > 0 && srand(env, seed)
    w = initweights(atype, hidden, N)
    opts = map(wi->deepcopy(optim), w)

    avgreward = 0
    bestE = +Inf
    for k=1:episodes
        state = reset!(env)
        episode_rewards = 0
        history = History(N, γ)
        for t=1:tmax
            # s = convert(atype, s)
            p, V = predict(w, state)
            action = sample_action(p)
            
            next_state, reward, done, _ = step!(env, action)
            append!(history.states, state)
            append!(history.actions, action)
            push!(history.rewards, reward)
            state = next_state
            episode_rewards += reward
            
            # k % infotime == 0 && rendered && render(env)
            done && break
        end

        avgreward = 0.01 * episode_rewards + avgreward * 0.99
        E = energy(env.J, env.σ)
        bestE = bestE < E ? bestE : E

        k % infotime == 0 && println("(episode:$k, avgreward:$(round(avgreward,3)),\tE:$(round(E,3)),\tbestE:$(round(bestE,3))")
        # k % infotime == 0 && rendered && render(env, close=true)

        train!(w, history, opts)
    end
    return w, env
end
