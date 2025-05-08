using LinearAlgebra

function rbf(x1, x2, ls)
    return exp(-norm(x1 - x2)^2 / (2 * ls^2))
end

function rbf_diff(x1, x2, ls)
    return rbf(x1, x2, ls) * norm(x1 - x2)^2 / ls^3
end

function rbf2(x1, x2, ls, os)
    return os * exp(-norm(x1 - x2)^2 / (2 * ls^2))
end

function rbf2_ds(x1, x2, ls, os)
    return os * rbf(x1, x2, ls) * norm(x1 - x2)^2 / ls^3
end

function rbf2_do(x1, x2, ls)
    return exp(-norm(x1 - x2)^2 / (2 * ls^2))
end



function kernel_matrix2(X, ls, σ², os)
    n = size(X, 1)
    K = zeros(n, n)
    for i = 1 : n
        for j = i : n
            K[i, j] = rbf2(X[i, :], X[j, :], ls, os)
            K[j, i] = K[i, j]
        end
    end

    K += σ² * I
    return K
end

function dmatrix2_ds(X, ls, os)
    n = size(X, 1)
    dKds = zeros(n, n)

    for i = 1 : n
        for j = i : n
            dKds[i, j] = rbf2_ds(X[i, :], X[j, :], ls, os)
            dKds[j, i] = dKds[i, j]
        end
    end
    return dKds
end

function dmatrix2_do(X, ls)
    n = size(X, 1)
    dKdo = zeros(n, n)

    for i = 1 : n
        for j = i : n
            dKdo[i, j] = rbf2_do(X[i, :], X[j, :], ls)
            dKdo[j, i] = dKdo[i, j]
        end
    end
    return dKdo
end










