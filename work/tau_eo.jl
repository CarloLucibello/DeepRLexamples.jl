using Distributions

"""
ground_state_τeo(g::AGraph, J::Vector{Vector{Int}}
    τ::Float64 = 1.2,
    maxiters::Int = typemax(Int),
    infotime::Int = typemax(Int))

Computes the ground state of an Ising model using the τ-EO heuristic.

Returns a configuration `σ` (a vector with ±1 components) and its energy `E`.
"""
function ground_state_τeo(g::AGraph, J::Vector{Vector{Int}};
    τ::Float64 = 1.2,
    σ0 = fill(1, nv(g)),
    maxiters::Int = typemax(Int),
    infotime::Int = typemax(Int))

    N = nv(g)
    σ = fill(1, N)
    hmax = maximum(v->sum(abs,v), J)
    sets = [Set{Int}() for s=1:2hmax+1]
    for i=1:N
        f = frust(g, σ, J[i], i)
        idx = ftoidx(f, hmax)
        push!(sets[idx], i)
    end
    distr = DiscreteDistribution(Float64[(τ-1)/(1-N^(1-τ)) / l^τ for l=1:N])
    E = energy(g, σ, zeros(Int, N), J)
    Emin = E
    σmin = deepcopy(σ)

    niter_tranche = 100
    itmultiplier = 4

    niters = 0
    foundnewmin = true

    while foundnewmin && niters <= maxiters
        foundnewmin = false
        for it=1:niter_tranche
            niters += 1
            niters > maxiters && break
            for _=1:N
                l = rand(distr)
                idx = searchsortedfirst(cumsum(length.(sets)), l)
                i = rand(sets[idx])

                delete!(sets[idx], i)
                for j in neighbors(g, i)
                    fj = frust(g, σ, J[j], j)
                    idxj = ftoidx(fj, hmax)
                    delete!(sets[idxj], j)
                end

                σ[i] *= -1
                fi = frust(g, σ, J[i], i)
                idx = ftoidx(fi, hmax)
                push!(sets[idx], i)
                for j in neighbors(g, i)
                    fj = frust(g, σ, J[j], j)
                    idxj = ftoidx(fj, hmax)
                    push!(sets[idxj], j)
                end

                E += 2fi
                if E < Emin && niters > 5 # avoids too much copying in the very first steps
                    foundnewmin = true
                    Emin = E
                    σmin = deepcopy(σ)
                end
            end
            if niters % infotime == 0
                println("it=$niters E/N=$(E/N)  Emin/N=$(Emin/N)")
            end
        end

        foundnewmin && (niter_tranche *= itmultiplier)
    end
    return σmin, Emin
end

ftoidx(f::Int, hmax::Int) = hmax - f + 1

function frust(g::AGraph, σ::Vector, J::Vector{Int}, i)
    H = 0
    for (k,j) in enumerate(neighbors(g, i))
        H += J[k] * σ[j]
    end
    return -H*σ[i]
end
