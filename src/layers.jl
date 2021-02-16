#
# Parrot.jl/src/layers.jl
#
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
#

"""
    Alsing(in::Integer, out::Integer)

Create a non-linear `Alsing` layer with trainable parameters `W`, `b`,
`α`, and `β`.

    y = (β .+ σ.(α .* (W*x .+ b)) .* (1 .- β)) .* (W*x .+ b)

The input `x` must be a vector of length `in`, or a batch of vectors
represented as an `in × N` matrix. The out `y` will be a vector or batch
of length `out`.

See [Alsing et al. (2020)](https://doi.org/10.3847/1538-4365/ab917f)
for more information about this activation function. Intended to be used for
emulating stellar population synthesis codes.

# Example
```
julia> a = Alsing(8, 3)
Alsing(8, 3)

julia> a(rand(8))
3-element Array{Float32,1}:
 0.9413336
 0.30788675
 0.5125884
```
"""
struct Alsing{S<:AbstractArray, T<:AbstractArray}
    W::S
    b::T
    α::T
    β::T
end

function Alsing(in::Integer, out::Integer;
                initW = Flux.glorot_uniform, initb = Flux.zeros,
                initα = Flux.glorot_uniform, initβ = Flux.glorot_uniform)
    return Alsing(initW(out,in), initb(out), initα(out), initβ(out))
end

Flux.@functor Alsing

function (a::Alsing)(x::AbstractArray)
    W, b, α, β = a.W, a.b, a.α, a.β
    (β .+ Flux.σ.(α .* (W*x .+ b)) .* (1 .- β)) .* (W*x .+ b)
end

function Base.show(io::IO, l::Alsing)
    print(io, "Alsing(", size(l.W, 2), ", ", size(l.W, 1), ")")
end

#
# Efficiency hack.
#
(a::Alsing{W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    invoke(a, Tuple{AbstractArray}, x)

(a::Alsing{W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    a(T.(x))


"""
    TransformPCA(P::MultivariateStats.PCA{AbstractFloat})
    TransformPCA(P::OnlineStats.CCIPCA)

Create a `TransformPCA` layer that uses Principal Component Analysis (PCA)
to transform its input into the PCA basis using a given PCA object `P`.

The input `x` must have dimensions compatible with `P` and the layer's output
will be of the dimensions given by the PCA transformation. No parameters are
trainable.

See also: [`ReconstructPCA`](@ref)
"""
struct TransformPCA{S<:AbstractArray, T<:AbstractArray}
    Pt::S
    μ::T
end

function TransformPCA(P::MultivariateStats.PCA{S}) where {S<:AbstractFloat}
    return TransformPCA(permutedims(P.proj), P.mean)
end

function TransformPCA(P::OnlineStats.CCIPCA)
    return TransformPCA(permutedims(P.U), P.center)
end

Flux.@functor TransformPCA

Flux.trainable(t::TransformPCA) = ()

function (t::TransformPCA)(x::AbstractArray)
    Pt, μ = t.Pt, t.μ
    Pt * (x .- μ)
end

function Base.show(io::IO, l::TransformPCA)
    print(io, "TransformPCA(", size(l.Pt, 2), ", ", size(l.Pt, 1), ")")
end

#
# Efficiency hack.
#
(t::TransformPCA{W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    invoke(t, Tuple{AbstractArray}, x)

(t::TransformPCA{W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    a(T.(x))


"""
    ReconstructPCA(P::MultivariateStats.PCA{AbstractFloat})
    ReconstructPCA(P::OnlineStats.CCIPCA)

Create a `ReconstructPCA` layer that uses Principal Component Analysis (PCA)
to reconstruct its input from the PCA basis using a given PCA object `P`.

The input `x` must have dimensions compatible with `P` and the layer's output
will be of the dimensions given by the PCA reconstruction. No parameters are
trainable.

See also: [`TransformPCA`](@ref)
"""
struct ReconstructPCA{S<:AbstractArray, T<:AbstractArray}
    P::S
    μ::T
end

function ReconstructPCA(P::MultivariateStats.PCA{S}) where {S<:AbstractFloat}
    return ReconstructPCA(P.proj, P.mean)
end

function ReconstructPCA(P::OnlineStats.CCIPCA)
    return ReconstructPCA(P.U, P.center)
end

Flux.@functor ReconstructPCA

Flux.trainable(r::ReconstructPCA) = ()

function (r::ReconstructPCA)(x::AbstractArray)
    P, μ = r.P, r.μ
    P * x .+ μ
end

function Base.show(io::IO, l::ReconstructPCA)
    print(io, "ReconstructPCA(", size(l.P, 2), ", ", size(l.P, 1), ")")
end

#
# Efficiency hack.
#
(r::ReconstructPCA{W})(x::AbstractArray{T}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    invoke(r, Tuple{AbstractArray}, x)

(r::ReconstructPCA{W})(x::AbstractArray{<:AbstractFloat}) where {T <: Union{Float32,Float64}, W <: AbstractArray{T}} =
    r(T.(x))


"""
    Normalize(μ::AbstractArray, σ::AbstractArray)

Create a simple `Normalize` layer with parameters consisting of
some previously known mean `μ` and standard deviation `σ` that are used
to normalize its input.

    y = (x .- μ) ./ σ

The input `x` must be a vector of equal length to both `μ` and `σ` or
an array with dimensions such that the broadcasted functions can be used
with the given `μ` and `σ`. No parameters are trainable.

See also: [`Denormalize`](@ref)
"""
struct Normalize{S<:AbstractArray}
    μ::S
    σ::S
end

function Normalize(μ::AbstractArray, σ::AbstractArray)
    return Normalize(μ, σ)
end

Flux.@functor Normalize

Flux.trainable(n::Normalize) = ()

function (n::Normalize)(x::AbstractArray)
    μ, σ = n.μ, n.σ
    (x .- μ) ./ σ
end

function Base.show(io::IO, l::Normalize)
    print(io, "Normalize(", size(l.μ, 1), ", ", size(l.μ, 1), ")")
end


"""
    Denormalize(μ::AbstractArray, σ::AbstractArray)

Create a simple `Denormalize` layer with parameters consisting of
some previously known mean `μ` and standard deviation `σ` that are used
to restore its input from a normalized state.

    y = (σ .* x) .+ μ

The input `x` must be a vector of equal length to both `μ` and `σ` or
an array with dimensions such that the broadcasted functions can be used
with the given `μ` and `σ`. No parameters are trainable.

See also: [`Normalize`](@ref)
"""
struct Denormalize{S<:AbstractArray}
    μ::S
    σ::S
end

function Denormalize(μ::AbstractArray, σ::AbstractArray)
    return Denormalize(μ, σ)
end

Flux.@functor Denormalize

Flux.trainable(d::Denormalize) = ()

function (d::Denormalize)(x::AbstractArray)
    μ, σ = d.μ, d.σ
    (σ .* x) .+ μ
end

function Base.show(io::IO, l::Denormalize)
    print(io, "Denormalize(", size(l.μ, 1), ", ", size(l.μ, 1), ")")
end

