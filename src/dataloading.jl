# Common data-loading routines go in here.
#import length from Base 
import Base.length, Base.getindex
using MLUtils

export shakespeare_ds

struct shakespeare_ds
    chars::Vector{Char} # Unique characters in dataest
    vocab_size::Int     # Number of unique characters
    split::Symbol       # Either 'train' or 'test'
    blocksize::Int      # Sequence length
    data::Vector{Int}   # Dataset
end

function shakespeare_ds(filename::String, split::Symbol, blocksize::Int)
    @assert split ∈ [:train, :test]
    # Load shakespeare dataset
    s = open(filename, "r") do io
        read(io, String)
    end
    chars = sort(unique(s))
    vocab_size = length(chars)

    # Develop a strategy to tokenize the text
    # i.e. convert the sequence of characters to sequence of integers
    # This is a simple example of a character-level encoder
    stoi = Dict([(c, i) for (c, i) ∈ zip(chars, 1:vocab_size)])

    # Build an encoder that maps characters present in the dataset to 
    # integers. These will be our tokens
    encode(s::String) = [stoi[c] for c ∈ s]

    # Split by train/test. Hard-code 80-20 split
    n_test = Int(round(length(s) * 0.8))
    @show n_test, length(s)
    if split == :train
        s = s[1:n_test]
    else 
        s = s[(n_test + 1):end]
    end
    # Encode the shakespeare dataset.
    data = encode(s);
    shakespeare_ds(chars, vocab_size, split, blocksize, data)
end

"""
    Builds a decoder that maps integers to the characters present in the dataset
"""
function get_decoder(ds::shakespeare_ds)
    itos = Dict([(i, c) for (c, i) ∈ zip(ds.chars, 1:ds.vocab_size)])
    decode(x::AbstractVector{Int}) = String([itos[i] for i ∈ x])
end

function get_encoder(ds::shakespeare_ds)
    encode(s::String) = [stoi[c] for c ∈ s]
end


"""
    Implement numobs and getobs methods for `shakespeare_ds` so that they 
    can be used to instantiate dataloaderss on top of them.
"""

# numobs just returns a large number. For the use case in this example,
# a dataloader for the shakespeare dataset will just fetch random batches.
# This should ensure that there are ample examples
numobs(ds::shakespeare_ds) = length(ds.data)

Base.length(ds::shakespeare_ds) = length(ds.data)

# getobs will pick a random sequence from the data.
# observations will be wrapped around the array

# function getobs(ds::shakespeare_ds, ix)
#     batch_size = length(ix)
#     reshape(view(ds.data, vcat([i:i+ds.blocksize-1 for i ∈ ix]...)), ds.blocksize, batch_size)
# end

function Base.getindex(ds::shakespeare_ds, ix)
    batch_size = length(ix)
    reshape(view(ds.data, vcat([i:i+ds.blocksize-1 for i ∈ ix]...)), ds.blocksize, batch_size)
end

function Base.getindex(ds::shakespeare_ds, ix::UnitRange{Int64})
    batch_size = length(ix)
    reshape(view(ds.data, vcat([i:i+ds.blocksize-1 for i ∈ ix]...)), ds.blocksize, batch_size)
end

#function get_dataloaders(blocksize, shuffle)




# End of file dataloading.jl