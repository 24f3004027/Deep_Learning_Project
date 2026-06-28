# ==================================================
# Introduction Directory Overview
# ==================================================

println("Initializing introduction documentation...")

# This directory contains introductory materials explaining 
# the deep learning concepts and practical use cases 
# related to this Context-Augmented Multiple-Choice QA project.

dir_contents = Dict(
    "an_intro_to_dl.txt"   => "A conceptual overview of deep learning.",
    "use_cases.jl"         => "Real-world applications of QA and RAG pipelines.",
    "about_this_dir.jl"    => "This directory guide."
)

for (file, desc) in dir_contents
    println(" - ", file, ": ", desc)
end
