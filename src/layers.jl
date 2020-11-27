#
# Parrot.jl/src/layers.jl
#
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
#

"""
    Normalization(μ::AbstractArray, σ::AbstractArray)

Create a simple `Normalization` layer with parameters consisting of
some previously known mean `μ` and standard deviation `σ`.

    y = (x .- μ) ./ σ

The input `x` must be a vector of equal length to both `μ` and `σ` or
an array with dimensions such that the broadcasted functions can be used
with the given `μ` and `σ`. No parameters are trainable.
"""
struct Normalization{S<:AbstractArray}
    μ::S
    σ::S
end

function Normalization(μ::AbstractArray, σ::AbstractArray)
    return Normalization(μ, σ)
end

Flux.@functor Normalization

Flux.trainable(n::Normalization) = ()


function (n::Normalization)(x::AbstractArray)
    μ, σ = n.μ, n.σ
    (x .- μ) ./ σ
end

function Base.show(io::IO, l::Normalization)
    print(io, "Normalization(", l.μ, ", ", l.σ, ")")
end

"""
    Alsing(in::Integer, out::Integer)

Create a non-linear `Alsing` layer with trainable parameters `W`, `b`,
`α`, and `β`.

    y = (β .+ σ.(α .* (W*x .+ b)) .* (1 .- β)) .* (W*x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors
represented as an `in × N` matrix. The out `y` will be a vector or batch
of length `out`.

See [arXiv:1911.11778](https://arxiv.org/abs/1911.11778) for more information
about this activation function. Intended to be used for emulating stellar
population synthesis codes.

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
                initW = Flux.glorot_uniform, initb = Flux.zeros,
                initα = Flux.glorot_uniform, initβ = Flux.glorot_uniform)
    return Alsing(initW(out,in), initb(out), initα(out), initβ(out))
end

Flux.@functor Alsing

function (a::Alsing)(x::AbstractArray)
    W, b, α, β = a.W, a.b, a.α, a.β
    (β .+ Flux.σ.(α .* (W*x .+ b)) .* (Flux.ones(size(β)) .- β)) .* (W*x .+ b)
end

function Base.show(io::IO, l::Alsing)
    print(io, "Alsing(", size(l.W, 2), ", ", size(l.W, 1), ")")
end

#
# This next bit of code is an optimization inspired by what Flux does for its
# Dense layer. While it's not entirely clear to me what this is doing, it
# apparently tells Julia to avoid generic matmul and instead hit more optimized
# matmul instead. I've also noticed this seems to force Float32 a bit more? Not
# a bad thing.
#
(a::Alsing{W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    invoke(a, Tuple{AbstractArray}, x)

(a::Alsing{W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    a(T.(x))

"""
    ReconstructPCA(P::MultivariateStats.PCA{AbstractFloat})

Create a `ReconstructPCA` layer that uses Principal Component analysis to
reconstruct its input using a given MutlivariateStats.PCA object `P`.

    y = MultivariateStats.reconstruct(P, x)

The input `x` must have dimensions compatible with `P` and the layer's output
will be of the dimensions given by the PCA reconstruction. No parameters are
trainable.
"""
struct ReconstructPCA{S<:AbstractFloat}
    P::MultivariateStats.PCA{S}
end

function ReconstructPCA(P::MultivariateStats.PCA{AbstractFloat})
    return ReconstructPCA(P)
end

Flux.@functor ReconstructPCA

Flux.trainable(r::ReconstructPCA) = ()

function (r::ReconstructPCA)(x::AbstractArray)
    P = r.P
    MultivariateStats.reconstruct(P,x)
end

function Base.show(io::IO, l::ReconstructPCA)
    print(io, "ReconstructPCA(", l.P, ")")
end


"""
    Denormalization(μ::AbstractArray, σ::AbstractArray)

Create a simple `Denormalization` layer with parameters consisting of
some previously known mean `μ` and standard deviation `σ`.

    y = (σ .* x) .+ μ

The input `x` must be a vector of equal length to both `μ` and `σ` or
an array with dimensions such that the broadcasted functions can be used
with the given `μ` and `σ`. No parameters are trainable.
"""
struct Denormalization{S<:AbstractArray}
    μ::S
    σ::S
end

function Denormalization(μ::AbstractArray, σ::AbstractArray)
    return Denormalization(μ, σ)
end

Flux.@functor Denormalization

Flux.trainable(d::Denormalization) = ()

function (d::Denormalization)(x::AbstractArray)
    μ, σ = d.μ, d.σ
    (σ .* x) .+ μ
end

function Base.show(io::IO, l::Denormalization)
    print(io, "Denormalization(", l.μ, ", ", l.σ, ")")
end

