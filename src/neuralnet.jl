#
# src/neuralnet.jl
#

greet() = print("Hello World!")

struct Alsing{S<:AbstractArray,T<:AbstractArray,U<:AbstractArray}
    W::S
    b::T
    β::U
    γ::U
end

function Alsing(in::Integer, out::Integer;
                initW = Flux.glorot_uniform, initb = zeros,
                initβ = Flux.glorot_uniform, initγ = Flux.glorot_uniform)
    return Alsing(initW(out,in), initb(out), initβ(out,out), initγ(out,out))
end

Flux.@functor Alsing

function (a::Alsing)(x::AbstractArray)
    W, b, β, γ = a.W, a.b, a.β, a.γ
    (γ .+ σ.(β * (W*x .+ b)) .* (one(γ) - γ)) * (W*x .+ b)
end

