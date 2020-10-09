push!(LOAD_PATH, "../src/")

using Documenter, Parrot

makedocs(
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
    ),
    sitename = "Parrot",
    pages = [
        "Home" => "index.md"
    ],
)

deploydocs(
    repo = "github.com/elijahmathews/Parrot.jl.git",
    devbranch = "primary",
)
