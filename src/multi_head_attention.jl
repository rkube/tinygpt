# Download and interpret the tiny shakespeare dataset
using Flux
using LinearAlgebra
using StatsBase
using Transformers
using Transformers.Basic
using Zygote

# block size is the length of tokens that are fed into the transformer at once
# When fed into the transformer, it means that there are block_size examples.
block_size = 8

# Number of sequences that are trained on in parallel
batch_size = 4
# Learning rate for optimization
lr = 1e-3
# Number of epochs to train on
num_epochs = 5_000
# Output interval
output_interval = 100
# Average loss in this window
avg_interval = 10

n_embed = 32  # Embedding dimension
head_size = n_embed # Set these equal for now


"""
    Data loading, tokenizer, pre-processing etc.
"""

s = open("datasets/tiny_shakespeare.txt", "r") do io
    read(io, String)
end

println("Length of dataset is $(length(s))")

# Get all unique characters that occur in the text
chars = sort(unique(s))
# Length of vocabulary
vocab_size = length(chars)

# Develop a strategy to tokenize the text
# i.e. convert the sequence of characters to sequence of integers
# This is a simple example of a character-level encoder
stoi = Dict([(c, i) for (c, i) ∈ zip(chars, 1:vocab_size)])
itos = Dict([(i, c) for (c, i) ∈ zip(chars, 1:vocab_size)])

# Build encoder and decoder
encode(s::String) = [stoi[c] for c ∈ s]
decode(x::AbstractVector{Int}) = String([itos[i] for i ∈ x])

# Encode the entire dataset
data = encode(s);

# Train-test split
n_test = Int(round(length(data) * 0.9))
data_train = data[1:n_test];
data_test = data[(n_test + 1):end];


function get_batch(split)
    data = split == "train" ? data_train : data_test
    ix = rand(1:length(data) - block_size, batch_size)
    # x has dimensions block_size, batch_size
    # dim1: features (individual tokens of a given sample)
    # dim2: samples (a feature across different samples)
    x = reshape(view(data_train, vcat([i:i+block_size-1 for i ∈ ix]...)), block_size, batch_size) 
    # y is the target. i.e. features shifted by 1
    y = reshape(view(data_train, vcat([i+1:i+block_size for i ∈ ix]...)), block_size, batch_size) 
    x, y
end

"""
    Embedding
"""
struct my_embed
    te::Embed
    pe::Embed
    block_size::Int
    vocab_size::Int
    n_embed::Int
end

# Make this a functor https://fluxml.ai/Functors.jl/stable/#Basic-Usage-and-Implementation
Flux.@functor my_embed

function my_embed(n_embed::Int, vocab_size::Int, block_size::Int)
    tt = Embed(n_embed, vocab_size)
    pp = Embed(n_embed, block_size)
    my_embed(tt, pp, block_size, vocab_size, n_embed)
end

"""
    Forward pass 
"""
function (e::my_embed)(idx)
    # Crop the input to the block size. 
    idx_cond = size(idx, 1) ≤ e.block_size ? idx : idx[end - e.block_size+1:end, :]
    te = e.te(idx_cond)           # Calculate character embedding
    pe = e.pe(axes(idx_cond, 1))  # Calculate positional embedding
    te .+ pe
end

# Test that embedding works. Embed some random tokens
e = my_embed(n_embed, vocab_size, block_size)
sample_ix = rand(1:vocab_size, 20, 1)
out = e(sample_ix) # size (C, T, B)

# Test if we get the correct parameters
ps = Flux.params(e)
length(ps) == 2
ps[1] == e.te.embedding
ps[2] == e.pe.embedding

"""
    Self-attention head

Stores a key, query and value. These are all just linear layers.
"""
struct self_attn
    key::Dense
    query::Dense
    value::Dense
end

# Explicitly define the trainable parameters. Bias are disabled by default
Flux.trainable(s::self_attn) = (s.key.weight, s.query.weight, s.value.weight)

#Flux.@functor(self_attn)

"""
    self_attn(n_embed, head_size)

Constructor that returns a new self-attention head.
"""
function self_attn(n_embed, head_size)
    key = Dense(n_embed, head_size, bias=false)
    query = Dense(n_embed, head_size, bias=false)
    value = Dense(n_embed, head_size, bias=false)
    self_attn(key, query, value)
end

"""
    Forward-pass for self-attention

Call this like

> my_sha = self_attn(32, 32)
> my_sha(out)
"""
function (sha::self_attn)(x)
    # Get size of data we operate on
    C, T, B = size(x)   # C: Embedding size, T=Sequence length, B=Batch size
    # Calculate key, query, value
    k = sha.key(x) # head_size, Token, batch
    q = sha.query(x)
    v = sha.value(x)
    # Perform causal self-attention
    # So far, no communication has happened. To get cross-affinity use batched multiplication from Transformers
    wts = Transformers.batchedmul(q, k, transA=true) ./ sqrt(1f0 * C)   
    # Now add the mask. Set the upper right triangular part to large negative values
    wts = wts .+ triu(ones(eltype(wts), T, T), 1) .* -1f10
    wts = softmax(wts; dims=2) # size (T, T, B)
    out = permutedims(Transformers.batchedmul(wts, v, transB=true), (2, 1, 3)) #
end


# test if output is valid
my_sha = self_attn(n_embed, n_embed)
my_sha(out)

# test if parameters are captured correctly
ps_sha = Flux.params(my_sha)
# The single-head attention model has 3 parameters: The weight of the key, query, and value mapping
ps_sha[1] == my_sha.key.weight
ps_sha[2] == my_sha.query.weight
ps_sha[3] == my_sha.value.weight

"""
    Multi-head attention
"""

struct multihead_attn
    heads
end

Flux.@functor multihead_attn
#Flux.trainable(s::self_attn) = (s.key.weight, s.query.weight, s.value.weight)


function multihead_attn(num_heads, head_size)
    hh = [self_attn(head_size, head_size ÷ num_heads) for _ ∈ 1:num_heads]
    multihead_attn(hh)
end

function (mha::multihead_attn)(x)
    # Apply each head in parallel to x. concatenate output.
    Parallel(vcat, mha.heads...)(x)
end

# Test if output works
my_mha = multihead_attn(4, n_embed)
my_mha(out)

size(my_mha(out)) == (32, 8, 1)

ps_mha = Flux.params(my_mha)
length(ps_mha) == 3 * length(my_mha.heads) # 3 parameters per self-attention head
ps_mha[1] == my_mha.heads[1].key.weight
ps_mha[2] == my_mha.heads[1].query.weight
ps_mha[3] == my_mha.heads[1].value.weight
ps_mha[4] == my_mha.heads[2].key.weight
ps_mha[8] == my_mha.heads[3].query.weight
ps_mha[12] == my_mha.heads[4].value.weight

model = Chain(my_embed(n_embed, vocab_size, block_size), 
              multihead_attn(4, n_embed),
              Dense(n_embed, vocab_size))

# Test if we the model accepts token sequences as input
sample_ix = rand(1:vocab_size, 20, 1)
model(sample_ix)


# Calculate loss. Should by -log(1/65)
#@show loss(xb, yb), -log(1.0 / 65.0)

# Now define a function that generates a new token.
# Input is an array of tokens, shape(T, B)
# Output is the same array of max_new_tokens newly sampled tokens, shape(T+max_new_tokens, B)
# Note that this function does not use the history of the sequence. We are extracting only the
# last time step (in logits[:, end, :]). In the future we have to change this.
function generate(idx, max_new_tokens)
    # idx is the array of indices in the current context.
    # size should be (T, B) T -> block_size (that is sequence length), B -> Batch size
    for dummy ∈ 1:max_new_tokens
        # crop idx to the last block_size tokens
        #idx_cond = size(idx, 1) < block_size ? idx : idx[end-block_size+1:end, :]  # size (T, B)
        logits = model(idx)   # size(logits) = (vocab_size, T, B)
        # Focus only on the last time step to predict the next token
        logits = logits[:, end, :]  # size(logits) = (vocab_size, 1, B)
        # Apply softmax to get probabilities
        probs = softmax(logits, dims=1)
        # Append a new token to idx. Do this for each batch
        idx_next = [sample(AnalyticWeights(probs[:, i])) for i ∈ axes(probs, 2)]
        idx = vcat(idx, idx_next')
    end
    return idx
end

# With this function we can now generate new output
idx = ones(Int, 1, 1)
decode(generate(idx, 20)[:, 1])


function loss(x, y)
    logits = model(x)
    Flux.Losses.logitcrossentropy(logits, Flux.onehotbatch(y, 1:vocab_size))
end

# Now we want to train this model
ps = Flux.params(model)
opt = ADAM(lr)

for epoch ∈ 1:num_epochs
    xb, yb = get_batch("train")
    grad = gradient(() -> loss(xb, yb), ps)
    Flux.update!(opt, ps, grad)
    # Estimate losses every output epochs
    if epoch % output_interval == 0
        for split ∈ ["train", "test"]
            loss_i = 0.0
            for i ∈ 1:avg_interval
                (xb, yb) = get_batch(split)
                loss_i += loss(xb, yb)
            end
            println("Epoch $(epoch): $(split): loss = $(loss_i / avg_interval)")
        end
    end
end

idx = ones(Int, 1, 1)
println("Output after training for $(num_epochs) epochs:")
print(decode(generate(idx, 500)[:, 1]))

# After ~5_000 epochs, we get a loss of about 2.3 - 2.4