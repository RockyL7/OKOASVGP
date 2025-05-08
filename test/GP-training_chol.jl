using LinearAlgebra
using SparseArrays
using Statistics
using Distributions
using MAT
using CSV
using DataFrames
using ArgParse
include("../src/load_data.jl")
include("../src/kernelfunc.jl")
include("../src/GP_chol.jl")

settings = ArgParseSettings()
@add_arg_table settings begin
    "--seed"
        help = "Random Seed"
        required = true
        arg_type = Int
    "--iter" 
        help = "Number of Optimization Steps"
        required = true
        arg_type = Int
    "--total_n"
        help = "Training Data Dimension"
        required = true 
        arg_type = Int
    "--α"
        help = "Learning Rate"
        required = true 
        arg_type = Float64
    "--sche"
        help = "Scheduler"
        required = true
        arg_type = String
    "--γ"
        help = "Scheduler Decay Factor"
        required = true 
        arg_type = Float64
    "--data"
        help = "Dataset"
        required = true
        arg_type = String
end
parsed_args = parse_args(settings)
seed = parsed_args["seed"]
total_n = parsed_args["total_n"]
iter = parsed_args["iter"]
## learning rate
α = parsed_args["α"]
## scheduler and decay factor
data = parsed_args["data"]
str = parsed_args["sche"]
split_str = split(str, ",")
trimmed_str = strip.(split_str)
sche = parse.(Float64, trimmed_str)
γ = parsed_args["γ"]

# @show seed
# @show iter
# @show total_n
# @show α
# @show sche
# @show γ
# @show data

## load the data
vars = matread("datasets2/$(data).mat")
train_x, train_y = load_data(vars["data"]; total_n = total_n, seed = seed)
n = size(train_x, 1)


# Random.default_rng() = MersenneTwister()

## set the initial guess of hyperparameters and form the kernel matrix
noise = 0.6932
lengthscale = 0.6931
outputscale = 0.6931
K = kernel_matrix2(train_x, lengthscale, noise, outputscale)



## generate the setting for GP learning
params = [[noise], [lengthscale], [outputscale]]
dLdθ = [[0.0], [0.0], [0.0]]
C = cholesky(K)
logdetK = 2*sum(log.(diag(C.U)))
sol = C.U \ (C.L \ train_y)
loss = 0.5 * (logdetK + train_y' * sol + n * log(2 * π)) / n
loss_list = zeros(iter+1)
loss_list[1] = loss
loss_list = GP_chol(K, train_x, train_y, iter, α, γ, sche, params, dLdθ, n, loss_list)

## generate the result CSV file
touch("dataframe/chol_$(iter)_$(α)_$(γ).csv")
efg = open("dataframe/chol_$(iter)_$(α)_$(γ).csv", "w")
mn = DataFrame(iter = 0:iter, 
            Loss = loss_list,
            ) 
CSV.write("dataframe/chol_$(iter)_$(α)_$(γ).csv", mn)
    











