# Read csv tensor from torch
# Compare crossentropy calculation to pytorch

# Corresponding python code:
"""
    import torch
    import torch.nn as nn
    import pandas as pd

    input = torch.randn(10, 5)
    df = pd.DataFrame(input.numpy())
    df.to_csv("input.csv", index=False)

    target = torch.empty(10, dtype=torch.long).random_(5)
    df = pd.DataFrame(target.numpy())
    df.to_csv("target.csv", index=False)

    loss = nn.CrossEntropyLoss()
    loss(input, target)
"""
#
using DelimitedFiles
using Flux

function array_from_csv(fname)

    input_csv = open(fname, "r") do io
        readlines(io)
    end
    readdlm(IOBuffer(join(input_csv[2:end], "\n")), ',', Float64)
end

target = Int64.(array_from_csv("../../tmp/target.csv"))
input = array_from_csv("../../tmp/input.csv")

target_oh = Flux.onehotbatch(target[:, 1], 0:4)
Flux.logitcrossentropy(input', target_oh)

