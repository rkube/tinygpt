# Download and interpret the tiny shakespeare dataset
using Flux
using StatsBase
using Transformers
using Transformers.Basic


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
data = encode(s)

# Train-test split
n_test = Int(round(length(data) * 0.9))
data_train = data[1:n_test]
data_test = data[(n_test + 1):end]

# block size is the length of tokens that are fed into the transformer at once
# When fed into the transformer, it means that there are block_size examples.
# 
block_size = 8
# These are `block-size` examples for the transformer
# 1. Input [19]: 48 is most likely the next token
# 2. Input [19, 48]: 57 is likely the next token
# 3. Input [19, 48, 57]: 58 is likely the next token
# ...
# 8. Input [19, 48, 57, 58, 59, 2, 16, 48]: 59 is likely the next token
# Spell this out in code:
x = data_train[1:block_size]
y = data_train[2:block_size+1]
for t ∈ 1:block_size
    context = x[1:t]
    target = y[t]
    println("For input $(context) the target is $(target)")
end

# Now introduce a batch size
batch_size = 4

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

xb, yb = get_batch("train")
for b ∈ 1:batch_size
    for t ∈ 1:block_size
        context = xb[1:t, b]
        target = yb[t, b]
        println("For input $(context) the target is $(target)")
    end
end

# Work with embeddings  
emb = Embed(vocab_size, vocab_size)

function enc_forward(x)
    # Assume that embedding gives the logits, which encode the probability (inverse of sigmoid)
    # of the next token.
    # Shape is [vocab_size, block_size, batch_size]
    emb(x)
end


function loss(x, y)
    logits = enc_forward(x)
    Flux.Losses.logitcrossentropy(logits, Flux.onehotbatch(y, 1:vocab_size))
end

# Calculate loss. Should by -log(1/65)
@show loss(xb, yb), -log(1.0 / 65.0)

# Now define a function that generates a new token.
# Input is an array of tokens, shape(T, B)
# Output is the same array of max_new_tokens newly sampled tokens, shape(T+max_new_tokens, B)
# Note that this function does not use the history of the sequence. We are extracting only the
# last time step (in logits[:, end, :]). In the future we have to change this.
function generate(idx, max_new_tokens)
    # idx is the array of indices in the current context.
    # size should be (T, B) T -> block_size (that is sequence length), B -> Batch size
    for dummy ∈ 1:max_new_tokens
        logits = enc_forward(idx)
        # Focus only on the last time step
        logits = logits[:, end, :]  # Now of shape vocab_size(C), B
        # Apply softmax to get probabilities
        probs = softmax(logits, dims=1)
        # a new token from the distribution. Do this for each batch
        idx_next = [sample(AnalyticWeights(probs[:, i])) for i ∈ axes(probs, 2)]
        idx = vcat(idx, idx_next')
    end

    return idx
end

# With this function we can now generate new output
idx = ones(Int, 1, 1)
decode(generate(idx, 20)[:, 1])

# Now we want to train this model

ps = Flux.params([emb])
opt = AdamW(1e-3)

@show emb.embedding
for epoch ∈ 1:10000
    xb, yb = get_batch("train")

    grad = gradient(() -> loss(xb, yb), ps)
    Flux.update!(opt, ps, grad)

    epoch % 1000 == 0 && @show loss(xb, yb)
end
@show emb.embedding

idx = ones(Int, 1, 1)
print(decode(generate(idx, 500)[:, 1]))

# After ~20_000 epochs, we get something reasonable-ish....