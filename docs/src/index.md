# Parrot
*A neural network framework for emulating stellar population synthesis. Written in [Julia](https://julialang.org).*

## Installation

First, please install Julia on your system by following the steps listed on the [Julia website](https://julialang.org/downloads/). This package is currently being developed and tested on Julia v1.6.

To install Parrot, open a Julia REPL and type `]` (to enter the Pkg REPL) followed by:

```
(@v1.6) pkg> add https://github.com/elijahmathews/Parrot.jl.git
```

Or alternatively, run the following code directly at the Julia REPL:

```julia
julia> import Pkg; Pkg.add(url="https://github.com/elijahmathews/Parrot.jl.git")
```

## Overview

At the moment, Parrot is in development, but the intent is to create a package that will aid in emulating stellar population synthesis (SPS) codes efficiently using neural networks. The package is currently being designed under the [Flux](https://github.com/FluxML/Flux.jl) machine learning ecosystem, and may be combined with [Turing](https://github.com/TuringLang/Turing.jl) probabilistic programming library for SED inference.

```@docs
Alsing
TransformPCA
ReconstructPCA
Normalize
Denormalize
```
