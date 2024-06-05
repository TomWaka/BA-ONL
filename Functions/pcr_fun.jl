module PCR_function

using MultivariateStats, Statistics, LinearAlgebra, Random, StatsBase

export PCR_predict, tune_num_components, PCR_logistic_predict, tune_num_components_logistic

# prediction via PCR
function PCR_predict(X_train, y_train, X_test, num_components)
    M = fit(PCA, X_train'; maxoutdim=num_components)
    X_train_reduced = transform(M, X_train')
    X_test_reduced = transform(M, X_test')
    β = X_train_reduced' \ y_train
    y_pred = X_test_reduced' * β
    return y_pred
end

# prediction via PCR for logistic regression
function PCR_logistic_predict(X_train, y_train, X_test, num_components)
    M = fit(PCA, X_train'; maxoutdim=num_components)
    X_train_reduced = transform(M, X_train')
    X_test_reduced = transform(M, X_test')
    β = logistic_regression(X_train_reduced', y_train)
    y_pred = 1 ./ (1 .+ exp.(-X_test_reduced' * β))
    return y_pred
end

function logistic_regression(X, y; max_iter=1000, lr=0.01)
    n, p = size(X)
    β = zeros(p)
    for iter in 1:max_iter
        y_pred = 1 ./ (1 .+ exp.(-X * β))
        gradient = X' * (y - y_pred) / n
        β += lr * gradient
    end
    return β
end

# tuning the number of principal components
function tune_num_components(X, y, max_components; n_folds=5)
    n_samples = size(X, 1)
    indices = collect(1:n_samples)
    shuffle!(indices) 
    
    fold_size = div(n_samples, n_folds)
    best_num_components = 1
    best_mse = Inf
    
    for num_components in 1:max_components
        mse_fold = []
        
        for i in 1:n_folds
            val_start = (i - 1) * fold_size + 1
            val_end = min(i * fold_size, n_samples)
            val_indices = indices[val_start:val_end]
            train_indices = setdiff(indices, val_indices)
            
            X_train, X_val = X[train_indices, :], X[val_indices, :]
            y_train, y_val = y[train_indices], y[val_indices]
            
            y_val_pred = PCR_predict(X_train, y_train, X_val, num_components)
            push!(mse_fold, mean((y_val .- y_val_pred).^2))
        end
        
        mean_mse = mean(mse_fold)
        if mean_mse < best_mse
            best_mse = mean_mse
            best_num_components = num_components
        end
    end
    
    return best_num_components
end

function tune_num_components_logistic(X, y, max_components; n_folds=5)
    n_samples = size(X, 1)
    indices = collect(1:n_samples)
    shuffle!(indices)
    
    fold_size = div(n_samples, n_folds)
    best_num_components = 1
    best_log_loss = Inf
    
    for num_components in 1:max_components
        log_loss_fold = []
        
        for i in 1:n_folds
            val_start = (i - 1) * fold_size + 1
            val_end = min(i * fold_size, n_samples)
            val_indices = indices[val_start:val_end]
            train_indices = setdiff(indices, val_indices)
            
            X_train, X_val = X[train_indices, :], X[val_indices, :]
            y_train, y_val = y[train_indices], y[val_indices]
            
            y_val_pred = PCR_logistic_predict(X_train, y_train, X_val, num_components)
            push!(log_loss_fold, -mean(y_val .* log.(y_val_pred) .+ (1 .- y_val) .* log.(1 .- y_val_pred)))
        end
        
        mean_log_loss = mean(log_loss_fold)
        if mean_log_loss < best_log_loss
            best_log_loss = mean_log_loss
            best_num_components = num_components
        end
    end
    
    return best_num_components
end

end # module