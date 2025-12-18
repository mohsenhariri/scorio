using Documenter
using Scorio

makedocs(
    sitename = "Scorio.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://mohsenhariri.github.io/scorio/julia/",
        edit_link = "main",
        assets = String[],
    ),
    modules = [Scorio],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ],
    repo = "https://github.com/mohsenhariri/scorio/blob/{commit}{path}#{line}",
)

# Note: We don't use deploydocs() here because deployment is handled
# by the unified GitHub Actions workflow that combines Python and Julia docs.
# The workflow uploads the built docs from docs/build/ to gh-pages branch.
