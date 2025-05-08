using LinearAlgebra
using SparseArrays
using Test
using Random
using Statistics
using Distributions
using AdaptivelySubsampledKrylov
using MAT
using CSV
using DataFrames
using ArgParse

settings = ArgParseSettings()
@add_arg_table settings begin
    "--seed"
        help = "Seed"
        required = true 
        arg_type = Int
    "--density"
        help = "Density"
        required = true 
        arg_type = Float64
end
parsed_args = parse_args(settings)
seed = parsed_args["seed"]
density = parsed_args["density"]
@show seed
@show density

Random.seed!(seed)
n = 500
tA1 = sprandn(n, n, density) + spdiagm(0=>fill(10.0, n))
A1 = (tA1'*tA1)

Random.seed!(seed)
right = rand(n)
b = right ./ norm(right)

sol1, iters1 = cg((x,y) -> mul!(x, A1, y), b, zeros(n); tol = 1e-8, maxIter = 2000)

para_list = []

push!(para_list, [(0.10, 0.0, 0.3), (0.10, 0.0, 0.0), (0.10, 3.0, 0.0), (0.10, 8.0, 0.0), (0.10, 13.0, 0.0)])
push!(para_list, [(0.05, 0.0, 0.5), (0.05, 0.0, 0.3), (0.05, 0.0, 0.1), (0.05, 0.0, 0.0), (0.05, 3.0, 0.0)])
# push!(para_list, [(0.02, 0.0, 0.5), (0.02, 0.0, 0.4), (0.02, 0.0, 0.3), (0.02, 0.0, 0.2), (0.02, 0.0, 0.1)])
# push!(para_list, [(0.10, 20.0, 0.0), (0.10, 32.0, 0.0), (0.10, 44.0, 0.0), (0.10, 56.0, 0.0), (0.10, 68.0, 0.0)])
# push!(para_list, [(0.05, 10.0, 0.0), (0.05, 22.0, 0.0), (0.05, 34.0, 0.0), (0.05, 46.0, 0.0), (0.05, 58.0, 0.0)])
# push!(para_list, [(0.02, 0.0, 0.4), (0.02, 0.0, 0.2), (0.02, 0.0, 0.0), (0.02, 12.0, 0.0), (0.02, 24.0, 0.0)])

compare_list = []
touch("dataframe/test_rrcg_same_seed$(seed)_density$(density).csv")
header = ["λ", "term_min", "init_prob", "variance", "iters"]
CSV.write("dataframe/test_rrcg_same_seed$(seed)_density$(density).csv", DataFrame([]); header = header)

num = 100000
total_num = 3 * num
# Random.default_rng() = MersenneTwister()
for (para) in para_list
    compare_mid = []
    for (λ, term_min, init_prob) in para
        result = zeros(5)
        result[1] = λ
        result[2] = term_min
        result[3] = init_prob
        norm_square_res_list = zeros(total_num)

        for i = 1 : total_num
            temp = zeros(n)
            temp_unweighted, temp, num_iters = rrcg((x,y) -> mul!(x, A1, y), b, temp; term_min = term_min, λ = λ, init_prob = init_prob, maxIter = n)
            norm_square_res_list[i] = dot(temp-sol1, A1*(temp - sol1))
            result[5] += num_iters
        end
        
        result[4] = sum(norm_square_res_list) / total_num
        result[5] /= total_num
        push!(compare_mid, result)
        row = [result[1] result[2] result[3] result[4] result[5]]
        CSV.write("dataframe/test_rrcg_same_seed$(seed)_density$(density).csv", DataFrame(row,:auto); append = true)
    end
    push!(compare_list, compare_mid)
end                 

println(compare_list)


