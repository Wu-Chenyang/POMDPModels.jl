# An N-dimensional light-dark problem, originally used to test MCVI
# A very simple POMDP with continuous state and observation spaces.
# maintained by @zsunberg

import Base: ==, +, *, -

"""
    LightDarkNDState

## Fields
- `loc`: position
- `status`: 0 = normal, negative = terminal
"""
struct LightDarkNDState{N}
    status::Int64
    loc::SVector{N, Float64}
end

LightDarkNDState(status, loc) = LightDarkNDState{length(loc)}(status, loc)

*(n::Number, s::LightDarkNDState) = LightDarkNDState(s.status, n*s.loc)

# The light source is at (5,5,...,5)
default_sigma(x) = fill(norm(x .- 5.)/sqrt(2 * length(x)) + 1e-2, length(x))

# Each dimension is independently observed
independent_sigma(x) = abs.(x .- 5.)./sqrt(2) .+ 1e-2

"""
    LightDarkND

An N-dimensional light dark problem. The goal is to be near [0,..,0]∈R^N. Observations are noisy measurements of the position.

Model
-----

   -3-2-1 0 1 2 3
...| | | | | | | | ...
          G   S

Here G is the goal. S is the starting location
"""
@with_kw struct LightDarkND{N,F<:Function} <: POMDPs.POMDP{LightDarkNDState{N},Int,Vector{Float64}}
    discount_factor::Float64 = 0.95
    correct_r::Float64 = 10.0
    incorrect_r::Float64 = -10.0
    step_size::Matrix{Float64} = Matrix(1.0I, N, N)
    movement_cost::Float64 = 0.0
    sigma::F = default_sigma
end

discount(p::LightDarkND) = p.discount_factor

isterminal(::LightDarkND, act::Int64) = act == 0

isterminal(::LightDarkND, s::LightDarkNDState) = s.status < 0


actions(::LightDarkND{N}) where N = -N:N


struct LDNormalNDStateDist{N}
    mean::SVector{N,Float64}
    std::SVector{N,Float64}
end

sampletype(::Type{LDNormalNDStateDist{N}}) where N = LightDarkNDState{N}
rand(rng::AbstractRNG, d::LDNormalNDStateDist{N}) where N = LightDarkNDState(0, d.mean + randn(rng, N).*d.std)
initialstate(pomdp::LightDarkND{N}) where N = LDNormalNDStateDist{N}(fill(2.0, N), fill(3.0, N))
initialobs(m::LightDarkND, s) = observation(m, s)

observation(p::LightDarkND, sp::LightDarkNDState) = MvNormal(sp.loc, p.sigma(sp.loc))

function transition(p::LightDarkND, s::LightDarkNDState{N}, a::Int) where N
    if a == 0
        return Deterministic(LightDarkNDState{N}(-1, s.loc))
    else
        sgn = sign(a)
        return Deterministic(LightDarkNDState{N}(s.status, s.loc .+ sgn .* p.step_size[sgn * a, :]))
    end
end

function reward(p::LightDarkND, s::LightDarkNDState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if norm(s.loc) < 1.0
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end

convert_s(::Type{A}, s::LightDarkNDState{N}, p::LightDarkND) where {N, A<:AbstractArray} = eltype(A)[s.status, s.loc...]
convert_s(::Type{LightDarkNDState{N}}, s::AbstractArray, p::LightDarkND) where N = LightDarkNDState{N}(Int64(s[1]), s[2:end])