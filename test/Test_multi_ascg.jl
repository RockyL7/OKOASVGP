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
tA2 = sprandn(n, n, density) + 0.8 * spdiagm(0=>fill(10.0, n))
A2 = (tA2'*tA2)

Random.seed!(seed)
tA3 = sprandn(n, n, density) + 1.3 * spdiagm(0=>fill(10.0, n))
A3 = (tA3'*tA3)

Random.seed!(seed)
right = rand(n)
b = right ./ norm(right)

sol1, iters1 = cg((x,y) -> mul!(x, A1, y), b, zeros(n); tol = 1e-8, maxIter = 2000)
sol2, iters2 = cg((x,y) -> mul!(x, A2, y), b, zeros(n); tol = 1e-8, maxIter = 2000)
sol3, iters3 = cg((x,y) -> mul!(x, A3, y), b, zeros(n); tol = 1e-8, maxIter = 2000)

para_list = [-0.1, 1.0, 6.0, 11.0, 16.0]
para_list = [-0.2, -0.1, 0.0, 3.0, 6.0]

compare_list = []
touch("dataframe/test_ascg_diff_seed$(seed)_density$(density).csv")
header = ["eta", "variance", "iters"]
CSV.write("dataframe/test_ascg_diff_seed$(seed)_density$(density).csv", DataFrame([]); header = header)

num = 300000
total_num = 3 * num
#Random.default_rng() = MersenneTwister()
for (η) in para_list
    result = zeros(3)
    result[1] = η
    norm_square_res_list = zeros(total_num)
    for i = 1 : num
        temp = zeros(n)   
        temp_unweighted, temp, num_iters = ascg((x,y) -> mul!(x, A1, y), b, temp; η = η)
        norm_square_res_list[i] = dot(temp - sol1, A1 * (temp - sol1))
        result[3] += num_iters 
    end
    for i = 1 : num  
        temp = zeros(n)
        temp_unweighted, temp, num_iters = ascg((x,y) -> mul!(x, A2, y), b, temp; η = η)
        norm_square_res_list[num + i] = dot(temp - sol2, A2 * (temp - sol2))
        result[3] += num_iters
    end
    for i = 1 : num
        temp = zeros(n)
        temp_unweighted, temp, num_iters = ascg((x,y) -> mul!(x, A3, y), b, temp; η = η)
        norm_square_res_list[2 * num + i] = dot(temp - sol3, A3 * (temp - sol3))
        result[3] += num_iters
    end
    result[2] = sum(norm_square_res_list) / (total_num)
    result[3] /= total_num 
    push!(compare_list, result)
    row = [result[1] result[2] result[3]]
    CSV.write("dataframe/test_ascg_diff_seed$(seed)_density$(density).csv", DataFrame(row,:auto); append = true)
end

println(compare_list)


