# Parrot
*A neural network framework for emulating stellar population synthesis. Written in [Julia](https://julialang.org).*

## Installation

First, please install Julia on your system by following the steps listed on the [Julia website](https://julialang.org/downloads/). This package is currently being developed and tested on Julia v1.5, but it will likely work with earlier versions as well.

To install Parrot, open a Julia REPL and type `]` (to enter the Pkg REPL) followed by:

```
(@v1.5) pkg> add https://github.com/elijahmathews/Parrot.jl.git
```

Or alternatively, run the following code directly at the Julia REPL:

```julia
julia> import Pkg; Pkg.add(url="https://github.com/elijahmathews/Parrot.jl.git")
```

## Overview

At the moment, Parrot is in development, but the intent is to create a package that will aid in emulating stellar population synthesis (SPS) codes using neural networks. The package is currently being designed under the [Flux](https://github.com/FluxML/Flux.jl) machine learning ecosystem.

```@docs
Alsing
ReconstructPCA
Normalization
Denormalization
```
