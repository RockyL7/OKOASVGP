using LinearAlgebra
using SparseArrays
using Test
using Random
using Statistics
using Distributions
# using Flux
# using Flux.Optimise: update!
# using ParameterSchedulers
# using ParameterSchedulers: Scheduler
using Flux, ParameterSchedulers
using Optimisers: adjust!
using MAT
using AdaptivelySubsampledKrylov



# function of GP training using RR-CG as the inner loop
function GP_rrcg(K::Matrix{Float64}, train_x::Matrix{Float64}, train_y::Vector{Float64}, iter::Int64, α::Float64, γ, sche, params, dLdθ, num_sample, n, loss_list, step_list, init_prob, λ, term_min)
    #Update the parameter by adam with scheduler 
    # s = Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter])
    # s_2 = Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter])
    # s_3 = Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter])
    s = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    s_2 = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    s_3 = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    # opt = Scheduler(s, Adam())
    # opt_2 = Scheduler(s_2, Adam())
    # opt_3 = Scheduler(s_3, Adam())
    opt = Adam()
    opt_2 = Adam()
    opt_3 = Adam()
    opt_st = Flux.setup(opt, params[1])
    opt_st_2 = Flux.setup(opt_2, params[2])
    opt_st_3 = Flux.setup(opt_3, params[3])
    sol = zeros(n)
    sol_2 = zeros(n)
    weighted_sol = zeros(n)
    weighted_sol_2 = zeros(n)
    total_step = 0
    i = 1
    params_prev = [params[1][1], params[2][1], params[3][1]]
    while (i<= iter)
        println(i)
        # Compute DK/dθ_1...DK/dθ_m
        dKds = dmatrix2_ds(train_x, params[2][1], params[3][1])
        dKdo = dmatrix2_do(train_x, params[2][1])
        # compute K^{-1}y
        sol, weighted_sol, step = rrcg((x, y)-> mul!(x, K, y), train_y, sol; term_min = term_min, λ = λ, init_prob = init_prob, maxIter = n)
        # println("step sol ", step)
        # flush(stdout)
        total_step += step
        for j = 1 : length(params) 
            #Compute the trace term by stochastic trace estimation
            e_trace = 0
            for k = 1 : num_sample
                #Random.seed!(seed * i)
                #probe_vec = rand(dis)
                #Random.seed!(100*i + 10*j + k)
                probe_vec = 2.0 * bitrand(n) .- 1.0
                sol_2, weighted_sol_2, step = rrcg((x, y)-> mul!(x, K, y), probe_vec, zeros(n); term_min = term_min, λ = λ, init_prob = init_prob, maxIter = n)
                # println("step trace ", step)
                # flush(stdout)
                if (j == 1)
                    trace = weighted_sol_2' * probe_vec
                elseif (j == 2)
                    trace = weighted_sol_2' * (dKds * probe_vec)
                else
                    trace = weighted_sol_2' * (dKdo * probe_vec)
                end
                e_trace += trace
                total_step += step
            end
            e_trace /= num_sample
            if (j == 1)
                dLdθ[j][1] = 1/2 * (e_trace - norm(weighted_sol)^2)
            elseif (j == 2)
                dLdθ[j][1] = 1/2 * (e_trace - weighted_sol' * (dKds * weighted_sol))
            else
                dLdθ[j][1] = 1/2 * (e_trace - weighted_sol' * (dKdo * weighted_sol))
            end
        end
        adjust!(opt_st, ParameterSchedulers.next!(s))
        adjust!(opt_st_2, ParameterSchedulers.next!(s_2))
        adjust!(opt_st_3, ParameterSchedulers.next!(s_3))
        opt_st, params[1] = Flux.update!(opt_st, params[1], dLdθ[1])
        opt_st_2, params[2] = Flux.update!(opt_st_2, params[2], dLdθ[2])
        opt_st_3, params[3] = Flux.update!(opt_st_3, params[3], dLdθ[3])
        if (params[1][1] < 0)
            params[1][1] = params_prev[1][1]
        end
        params_prev = [params[1][1], params[2][1], params[3][1]]
        # Update the new kernel matrix
        K = kernel_matrix2(train_x, params[2][1], params[1][1], params[3][1])
        # Compute the updated loss function
        C = cholesky(K)
        logdetK = 2*sum(log.(diag(C.U)))
        sol_chol = C.U \ (C.L \ train_y)
        loss_list[i+1] = 0.5 * (logdetK + train_y' * sol_chol + n * log(2 * π)) / n
        println("Loss:", loss_list[i+1])
        flush(stdout)
        step_list[i+1] = total_step
        i += 1
    end
    return loss_list, step_list
end