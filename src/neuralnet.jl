# 
# Parrot.jl/src/neuralnet.jl
# 
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
# 

"""
    Alsing(in::Integer, out::Integer)

Create a non-linear `Alsing` layer with parameters `W`, `b`, `α`, and `β`.

    y = (β .+ σ.(α .* (W*x .+ b)) .* (1 .- β)) .* (W*x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors represented
as an `in × N` matrix. The out `y` will be a vector or batch of length `out`.

See [arXiv:1911.11778](https://arxiv.org/abs/1911.11778) for more information.

# Example
```
julia> a = Alsing(8,3)
Alsing(8, 3)

julia> a(rand(8))
3-element Array{Float64,1}:
  0.9555069887698046
  0.14443222298921962
 -0.06765412728860333
```
"""
struct Alsing{S<:AbstractArray, T<:AbstractArray, U<:AbstractArray}
    W::S
    b::T
    α::U
    β::U
end

function Alsing(in::Integer, out::Integer;
                initW = Flux.glorot_uniform, initb = zeros,
                initα = Flux.glorot_uniform, initβ = Flux.glorot_uniform)
    return Alsing(initW(out,in), initb(out), initα(out), initβ(out))
end

Flux.@functor Alsing

function (a::Alsing)(x::AbstractArray)
    W, b, α, β = a.W, a.b, a.α, a.β
    (β .+ σ.(α .* (W*x .+ b)) .* (1 .- β)) .* (W*x .+ b)
end

function Base.show(io::IO, l::Alsing)
    print(io, "Alsing(", size(l.W, 2), ", ", size(l.W, 1), ")")
end

