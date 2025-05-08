using LinearAlgebra
using SparseArrays
using Test
using Random
using Statistics
using Distributions
using MAT
using CSV
using DataFrames
using ArgParse
include("../src/load_data.jl")
include("../src/kernelfunc.jl")
include("../src/GP_cg.jl")

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
    "--num_sample"
        help = "Number of Probe Vectors"
        required = true 
        arg_type = Int
    "--trial"
        help = "Trial"
        required = true 
        arg_type = Int
    "--data"
        help = "Dataset"
        required = true
        arg_type = String
    "--tol"
        help = "Tolerence Value"
        required = true 
        arg_type = Float64
    "--max_iter"
        help = "max cg iteration"
        required = false 
        arg_type = Int
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
## number of probe vectors
num_sample = parsed_args["num_sample"]
## generate the setting for CG
trial = parsed_args["trial"]
tol = parsed_args["tol"]

## load the data
vars = matread("datasets2/$(data).mat")
train_x, train_y = load_data(vars["data"]; total_n = total_n, seed = seed)
n = size(train_x, 1)

if parsed_args["max_iter"] !== nothing
    max_iter = parsed_args["max_iter"]
else
    println("Warning: 'max_iter' not provided, using default value.")
    max_iter = n  # Set a default value if needed
end
# @show seed
# @show iter
# @show total_n
# @show α
# @show sche
# @show γ
# @show num_sample
# @show trial
# @show data
# @show tol
# @show max_iter

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
step_list = zeros(iter+1)
loss_list[1] = loss
step_list[1] = 0
loss_list, step_list = GP_cg(K, train_x, train_y, iter, α, γ, sche, params, dLdθ, num_sample, n, loss_list, step_list, tol, max_iter)

## generate the result CSV file
touch("dataframe/$(data)-$(trial)/cg_$(iter)_$(num_sample)_$(α)_$(γ)_$(tol)_$(max_iter).csv")
efg = open("dataframe/$(data)-$(trial)/cg_$(iter)_$(num_sample)_$(α)_$(γ)_$(tol)_$(max_iter).csv", "w")
mn = DataFrame(iter = 0:iter, 
            Loss = loss_list,
            Step = step_list
            ) 
CSV.write("dataframe/$(data)-$(trial)/cg_$(iter)_$(num_sample)_$(α)_$(γ)_$(tol)_$(max_iter).csv", mn)


