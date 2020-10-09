# Parrot
*A Neural Net Emulator for Prospector*

## Installation

First, please install Julia on your system by following the steps listed on the [Julia website](https://julialang.org/downloads/). This package is currently being developed and tested on Julia v1.5, but it will likely work with earlier versions as well.

Next, clone the repository and start a Julia shell within (once the repo is public, this will be possible to do from within Julia itself):

```bash
$ git clone git@github.com:elijahmathews/Parrot.jl.git
$ cd Parrot.jl
$ julia
```

Next, activate the package:
```julia
julia> cd("/path/to/Parrot.jl")
julia> using Pkg; Pkg.activate("."); using Parrot;
```

Then you can run the one function it currently contains:
```jldoctest greet; setup = :(using Parrot)
julia> Parrot.greet()
Hello World!
```
