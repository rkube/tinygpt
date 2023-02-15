using LinearAlgebra
using Random
using Statistics
using Transformers

# Gist of this lesson:
# We can do weighted aggregation of past elements by using matrix multiplication

# Manual seed
Random.seed!(1337)
B, T, C = 4, 8, 2 # batch, time (sequence length), channels (embedding dimension)
x = randn(C, T, B)

# We got T=8 tokens and would like to get them talking to each other
# i.e. couple the.
# C is some information within the sequence. Now the features at each
# index in time are two-dimensional. This extends the previous case where
# we only had the embedding at each time index.

# time: 44:30
# Couple them in a causal way. Token at position i should only couple to tokens 1...i-1
# A simple way to propagate information on previous tokens to the current one is
# the average value of all previous tokens.

# First we write this mean calculation in an explicit way.
# Build tensor with x[t, b] = mean_{i <= t} x[i, b]
xbow = zeros(C, T, B) # bag-of-words
for ix_b ∈ axes(xbow, 3)
    for ix_t ∈ axes(xbow, 2)
        # Inner loop averages over all previous features at the current time-step.
        # This is done independently for each batch.
        xprev = x[:, 1:ix_t, ix_b]
        xbow[:, ix_t, ix_b] = mean(xprev, dims=2)
    end  
end
@show xbow


# We can implement this averaging by matrix multiplication
# A is the matrix that has the coefficients for running average
Random.seed!(42)
A = tril(ones((3, 3)))  # Lower left triangular matrix. Has i ones in the i-th row.
A = A ./ sum(A, dims=2)  # Divide by the sum in each row
@show A

B = Float32.(rand(1:10, (3,2)))

A*B

# The columns of A*B now contain the running average along the column direction.
# That is, each row in A*B contains the average of the rows in B.

# Let's vectorize the weighted sum operation above:

wts = tril(ones(T, T))
wts = wts ./ sum(wts, dims=2)
# For GPU issues: https://discourse.julialang.org/t/how-to-broadcast-or-batch-multiply-a-batch-of-matrices-with-another-matrix-on-the-gpu/67259
# Or look at reshaping code in Flux.Dense: https://github.com/FluxML/Flux.jl/blob/c5a691aa3e74c0474e4ad4eb135800b45524bec4/src/layers/basic.jl#L176
xbow2 = mapslices(slice -> (wts * slice')', x, dims=(1, 2))

############################################################################################################
# A different method to obtain the weight matrix is to use softmax:
wts2 = zeros(Float32, T, T)
wts2[tril(ones(T, T)) .== 0] .= -1f10
wts2 = softmax(wts2; dims=2)
wts2 ≈ wts

############################################################################################################
## Version 4: self-attention
Random.seed!(42)
B, T, C = 4, 8, 32
x = randn(Float32, C, T, B) # (embedding, block_size, batch_size)

# Every token at each position emitts 2 vectors
# query - what am I looking for
# key - what do I contain
# Q*K gives encodes affinity between tokens

# imeplement a singel self-attention head
head_size = 16
key = Dense(C, head_size, bias=false)
query = Dense(C, head_size, bias=false)
value = Dense(C, head_size, bias=false)

k = key(x) # (head_size, T, B)
q = query(x) # (head_size, T, B)
v = value(x) # (head_size, T, B)
# So far, no communication has happened. To get cross-affinity use batched multiplication from Transformers
wts3 = Transformers.batchedmul(q, k, transA=true)
# Test output of batched matrix multiplication
wts3[:,:,1] ≈ q[:,:,1]' * k[:,:,1]

wts3[tril(ones(T, T)) .== 0, :] .= -1f10
wts3 = softmax(wts3; dims=2) # size (T, T, B)

out = permutedims(Transformers.batchedmul(wts3, v, transB=true), (2, 1, 3)) # size(head_size, T, B)
# Output of batch matrix multiplication:
wts3[:,:,2] * v[:,:,2]' ≈ out[:,:,2]'