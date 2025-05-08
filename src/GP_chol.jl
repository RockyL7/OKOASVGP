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

# function of GP training using Cholesky as the inner loop
function GP_chol(K::Matrix{Float64}, train_x::Matrix{Float64}, train_y::Vector{Float64}, iter::Int64, α::Float64, γ, sche, params, dLdθ, n, loss_list)
    #Update the parameter by adam with scheduler
    s = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    s_2 = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    s_3 = ParameterSchedulers.Stateful(Step(λ = α, γ = γ, step_sizes = [sche[1] * iter, sche[2] * iter]))
    # opt = Scheduler(Adam(), s)
    # opt_2 = Scheduler(Adam(), s_2)
    # opt_3 = Scheduler(Adam(), s_3)
    opt = Adam()
    opt_2 = Adam()
    opt_3 = Adam()
    opt_st = Flux.setup(opt, params[1])
    opt_st_2 = Flux.setup(opt_2, params[2])
    opt_st_3 = Flux.setup(opt_3, params[3])
    params_prev = [params[1][1], params[2][1], params[3][1]]
    Id = zeros(n, n) + I
    # compute K^{-1}y
    C = cholesky(K)
    mid = C.L \ train_y
    sol = C.U \ mid
    i = 1
    while (i<=iter)
        println(i)
        # Compute DK/dθ_1...DK/dθ_m
        dKds = dmatrix2_ds(train_x, params[2][1], params[3][1])
        dKdo = dmatrix2_do(train_x, params[2][1])
        # Compute the trace term by cholesky
        M = C.U \ Id
        K_inv = M * M'
        for j = 1 : length(params)
            if (j ==1)
                dLdθ[j][1] = 1/2 * (tr(K_inv) - sol' * sol)
            elseif (j == 2)
                dLdθ[j][1] = 1/2 * (tr(K_inv * dKds) - sol' * (dKds * sol))
            else
                dLdθ[j][1] = 1/2 * (tr(K_inv * dKdo) - sol' * (dKdo * sol))
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
        logdetK = 2 * sum(log.(diag(C.U)))
        mid = C.L \ train_y
        sol = C.U \ mid
        loss_list[i+1] = 0.5 * (logdetK + train_y' * sol + n * log(2 * π)) / n
        println("Loss:", loss_list[i+1])
        flush(stdout)
        i += 1
    end
    return loss_list
end