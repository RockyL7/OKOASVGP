using MAT
using Statistics


function load_data(dataset; total_n = -1, train_p = 0.64, seed = 14)
    X = dataset[:, 1:end-1]
    y = dataset[:, end]
    # remove dims with no variance
    good_dims = var(X, dims=1) .> 1.0e-10
    X = X[:, vec(good_dims)]

    # shuffling 
    Random.seed!(seed)
    shuffle_ind = randperm(size(X, 1))
    X = X[shuffle_ind, :]
    y = y[shuffle_ind]

    if total_n != -1
        X = X[1:total_n, :]
        y = y[1:total_n, :]
    end
    X = X .- minimum(X[:, 1])
    X = 2.0 * (X ./ maximum(X[:, 1])) .- 1.0
    y = y .- mean(y)
    y = y ./ std(y)

    train_n = convert(Int, floor(train_p * size(X, 1)))
    # valid_n = convert(Int, floor(valid_p * size(X, 1)))
    
    train_x = X[1 : train_n, :]
    train_y = y[1 : train_n]

    return train_x, train_y
end

# function split_data(x, y, train_n, valid_n)
#     train_x = x[1 : train_n, :]
#     train_y = y[1 : train_n]
#     # valid_x = x[train_n+1 : train_n+valid_n, :]
#     # valid_y = y[train_n+1 : train_n+valid_n]
#     # test_x = x[train_n+valid_n : end, :]
#     # test_y = y[train_n+valid_n : end]

#     return train_x, train_y #, valid_x, valid_y, test_x, test_y
# end


