module Softplus_function

using LinearAlgebra, Statistics, Random

export VI_softplus,  Softplus

# Swish function
# function Swish(a)
#     return a/(1+exp(-a))
# end    
    
# derivative of Swish 
# function DerSwish(a)
#     return (exp(a) * (1 + exp(a) + a)) / (1 + exp(a))^2
# end   

# Softplus (SmoothReLU function)
function Softplus(a)
    return log(1+exp(a))
end    


# derivative of Softplus
function DerSoft(a)
    return 1 / (1 + exp(-a))
end   
    
function DerLikelihood(X, beta_tilde, y, mini_batch_indices, B)
    a = X[mini_batch_indices, :] * beta_tilde
    #exp_a = exp.(a) 
    #swish_a = Swish.(a) #a .* exp_a ./ (1 .+ exp_a)
    #der_swish_a = DerSwish.(a) # exp_a .* (1 .+ exp_a .+ a) ./ (1 .+ exp_a).^2
    soft_a =  Softplus.(a)
    der_soft_a = DerSoft.(a)
    dif_likelihood = B * ((y[mini_batch_indices] - soft_a) .* der_soft_a)' * X[mini_batch_indices, :]
    return dif_likelihood'
end
    
# function for variational inference and adam
function VI_softplus(y, X, max_iter=100000, alpha_=0.0003, beta1=0.9, beta2=0.999, epsilon=1e-8)
        n, p = size(X)
        X_centered = X .- mean(X, dims=1)  # Center the data
        D_est = transpose(X_centered) * X_centered / n  # Estimate covariance matrix
        diag_est = eigen(D_est)  # Eigen decomposition of estimated covariance
        k_est = round(Int, log(n))  # Estimate rank
        # Inverse of estimated covariance matrix using eigen decomposition
        D_inv = diag_est.vectors[:, p-k_est+1:p] * Diagonal(1 ./ diag_est.values[p-k_est+1:p]) * transpose(diag_est.vectors[:, p-k_est+1:p])
        D_inv = (D_inv + D_inv') / 2  # Ensure symmetry
    
        # Initialization
        mu = zeros(p)  # Initialize mean vector
        m_mu = v_mu = zeros(p)  # First and second moments of mu
        CholCov = ones(p)/2  # Initialize covariance as identity matrix
        m_CholCov = v_CholCov = zeros(p)  # First and second moments of CholCov
        B = n / 50  # Mini-batch size
    
        for i in 1:max_iter
            # Compute gradients
            ep = randn(p)  # Random vector for stochastic estimation
            mini_batch_indices = randperm(n)[1:Int(n/B)]  # Mini-batch indices
            beta_tilde = mu + CholCov .* ep  # Perturb parameters for gradient estimation
            dif_prior = D_inv * beta_tilde  # Gradient-prior
            # Gradient-likelihood (using mini-batch)
            dif_likelihood = DerLikelihood(X, beta_tilde, y, mini_batch_indices, B)
            mu_grad = dif_prior - dif_likelihood  # Gradient w.r.t. mu
            cov_grad = - 1 ./ CholCov + mu_grad .* ep  # Gradient w.r.t. covariance
    
            # # Update first and second moments
            m_mu = beta1 .* m_mu + (1. - beta1) .* mu_grad
            v_mu = beta2 .* v_mu + (1. - beta2) .* mu_grad.^2
            m_CholCov = beta1 .* m_CholCov + (1. - beta1) .* cov_grad
            v_CholCov = beta2 .* v_CholCov + (1. - beta2) .* cov_grad.^2
    
            m_mu_hat = m_mu / (1. - beta1^i)
            v_mu_hat = v_mu / (1. - beta2^i)
            m_CholCov_hat = m_CholCov ./ (1. - beta1^i)
            v_CholCov_hat = v_CholCov ./ (1. - beta2^i)
    
            # Parameter update
            mu -= alpha_ .* m_mu_hat ./ ( sqrt.(v_mu_hat) .+ epsilon)    # Update mean vector
            CholCov -= alpha_*10 * m_CholCov_hat ./ ( sqrt.(v_CholCov_hat) .+ epsilon)    # Update covariance
        end
    return mu, CholCov  # Return the estimated parameters
end

end