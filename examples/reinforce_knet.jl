using Knet
using Gym
using DeepRL
using ArgParse
import AutoGrad: getval

function main(args)
    s = ArgParseSettings()
    s.description = ""

    @add_arg_table s begin
        ("--hidden"; default=[100]; nargs='*'; arg_type=Int64)
        ("--atype"; default=(gpu()>=0 ? "KnetArray{Float32}":"Array{Float64}"))
        ("--optim"; default="Adam(;lr=1e-4)")
        ("--discount"; default=0.95; arg_type=Float64)
        ("--episodes"; default=1000; arg_type=Int64)
        ("--seed"; default=-1; arg_type=Int64)
        ("--period"; default=100; arg_type=Int64)
    end
    rendered = true

    isa(args, String) && (args = split(args))
    o = parse_args(args, s; as_symbols=true); println(o)
    o[:atype] = eval(parse(o[:atype]))
    o[:seed] > 0 && srand(o[:seed])

    threshold =  500
    env = GymEnv("CartPole-v1")
    w = initweights(o[:atype], o[:hidden])
    opts = map(wi->eval(parse(o[:optim])), w)

    avgreward = 0
    for k = 1:o[:episodes]
        state = reset!(env)
        episode_rewards = 0
        history = EntryHist[]
        for t=1:1000
            s = convert(o[:atype], state)
            pred = predict(w,s)
            probs = softmax(pred)
            action = sample_action(probs)

            next_state, reward, done, _ = step!(env, actions(env)[action])
            push!(history, EntryHist(state, action, reward))
            state = next_state
            episode_rewards += reward

            k % o[:period] == 0 && rendered && (render(env); sleep(0.05))
            done && break
        end

        avgreward = 0.01 * episode_rewards + avgreward * 0.99

        k % o[:period] == 0 && println("(episode:$k, avgreward:$avgreward)")
        k % o[:period] == 0 && rendered && render(env, close=true)

        values = get_values(history, o[:discount]; threshold=threshold)
        for t=1:length(history)
            state = history[t].state
            action = history[t].action
            value = values[t]
            train!(w, state, action, value, opts)
        end
    end
end

struct EntryHist
    state
    action
    reward
end

function initweights(atype, hidden, xsize=4, ysize=2, winit=0.1)
    w = []
    x = xsize
    for y in [hidden..., 2]
        push!(w, xavier(y, x))
        push!(w, zeros(y, 1))
        x = y
    end
    return map(wi->convert(atype,wi), w)
end

function predict(w,x)
    for i = 1:2:length(w)-2
        x = relu.(w[i] * x .+ w[i+1])
    end
    return w[end-1] * x .+ w[end]
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

function get_values(history, γ; threshold=500)
    T = typeof(history[1].reward)
    values = Array{T}(length(history)+1)
    if length(history) < threshold
        values[end] = length(history) - threshold
    else
        values[end] = 0
    end
    for k = length(history):-1:1
        values[k] = γ * values[k+1] + history[k].reward
    end
    return values[1:end-1]
end

function loss(w, x, actions, values)
    ypred = predict(w,x)
    nrows, ncols = size(ypred)
    index = actions + nrows*(0:(length(actions)-1))
    lp = logp(ypred, 1)[index]
    return -sum(lp .* values) / size(x, 2)
end

function train!(w, s, a, v, opt)
    g = grad(loss)(w, s, a, v)
    update!(w, g, opt)
end

!isinteractive() && main(ARGS)
