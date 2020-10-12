#
# src/neuralnet.jl
#

greet() = print("Hello World!")

struct Alsing{S<:AbstractArray,T<:AbstractArray}
    W::S
    b::T
    β::S
    γ::S
end

function Alsing(in::Integer, out::Integer;
		initW = Flux.glorot_uniform, initb = zeros,
		initβ = Flux.glorot_uniform, initγ = Flux.glorot_uniform)
    return Alsing(initW(out,in), initb(out), initβ(out,in), initγ(out,in))
end

Flux.@functor Alsing

function (a::Alsing)(x::AbstractArray)
    W, b, β, γ = a.W, a.b, a.β, a.γ
    (γ + σ.(β*x) * (1-γ)) * (W*x + b)
end

