using Documenter
using Scorio

makedocs(
    sitename = "Scorio.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Scorio],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ]
)

deploydocs(
    repo = "github.com/mohsenhariri/scorio.git",
    devbranch = "main",
    push_preview = true
)
