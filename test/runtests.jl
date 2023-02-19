#
using tiny_gpt
using Test
using MLUtils

# Instantiate a dataset
@testset "tinygpt.jl" begin
    @testset "dataloading" begin
        ds_train = shakespeare_ds("../datasets/tiny_shakespeare.txt", :train, 16);
        ds_test = shakespeare_ds("../datasets/tiny_shakespeare.txt", :test, 16);

        # These are the lengths for a 80-20 test-train split
        @test size(ds_train.data) == (892315, )
        @test size(ds_test.data) == (223079, )
        # Vocabulary size should be identical
        @test ds_train.vocab_size == ds_test.vocab_size

        # Test if the dataloaders get the first few tokens correct.
        loader_train = DataLoader(ds_train, batchsize=1, shuffle=false)
        loader_test = DataLoader(ds_test, batchsize=1, shuffle=false)
        @test first(loader_train) == reshape([19; 48; 57; 58; 59; 2; 16; 48; 59; 48; 65; 44; 53; 11; 1; 15], 16, 1)
        @test first(loader_test) == reshape([64; 54; 60;  2; 40; 57; 44;  7;  1; 33; 47; 40; 59;  2; 48; 58], 16, 1)

        # Test shape for larger batch size
        loader_train = DataLoader(ds_train, batchsize=9, shuffle=false)
        xb = first(loader_train)
        @test size(xb) == (16, 9)
        @test xb[2:end, 1] == xb[1:end-1, 2]

    end
end
