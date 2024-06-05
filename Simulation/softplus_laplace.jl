using LinearAlgebra, Distributions, Random, SparseArrays, Plots, Statistics

include("Functions/softplus_fun.jl")
using .Softplus_function # module

include("Functions/pcr_fun.jl")
using .PCR_function # module

# function for laplace distribution
function MVL(mean, covariance)
    x = rand(MvNormal(mean, covariance))
    W = rand(Exponential(1))
    return sqrt.(W) .* x
end

# function for numerical simulation
function simulate(n, iteration)
    rate = 4/3
    p = round(Int, n^rate)
    D = 10 * exp.(- (1:p) / 7) .+ n * exp(-n^(1/3)) / p
    U = qr(randn(p, p)).Q  # an orthonormal matrix
    Sigma = Diagonal(D)
    mvn = MvNormal(zeros(p), Sigma)
    theta = randn(p) 
    X = (U *  hcat([MVL(zeros(p), Sigma) for _ in 1:n]...))'  # laplace distribution
    y = Softplus.(X * theta) .+ randn(n) 
    
    # perform estimation
    mu, CholCov = VI_softplus(y, X, 150000, 0.0003)
    CholCov = Diagonal(CholCov)

    max_components = min(size(X, 2), 30)  # set the upper limit to 30
    best_num_components = tune_num_components(X, y, max_components)

    # generate test data
    X_test = (U *  hcat([MVL(zeros(p), Sigma) for _ in 1:5000]...))'
    y_true = Softplus.(X_test * theta) 

    # prediction 
    posterior_sample = hcat([rand(MvNormal(mu, CholCov*CholCov')) for _ in 1:2000]...)
    posterior_prediction = Softplus.(X_test * posterior_sample)
    y_pred = mapslices(median, posterior_prediction; dims=2)
    posterior_upper = mapslices(row -> quantile(row, 0.975), posterior_prediction; dims=2)
    posterior_lower = mapslices(row -> quantile(row, 0.025), posterior_prediction; dims=2)

    # prediction via pcr
    y_pred_pcr = PCR_predict(X, y, X_test, best_num_components)
    MSE_pcr = sqrt(mean((y_true .- y_pred_pcr).^2))

    # calculate RMSE and 
    MSE = sqrt(mean((y_true .- y_pred).^2))
    CP = mean(posterior_upper .>= y_true .>= posterior_lower)
    AL = mean(posterior_upper .- posterior_lower)
    MSE_pcr = sqrt(mean((y_true .- y_pred_pcr).^2))
    return MSE, CP, AL, MSE_pcr
end

results = Dict()
for n in 50:50:500
    results[n] = Dict("MSE" => [], "CP" => [], "AL" => [], "MSE_pcr" => [])
    for iteration in 1:20
        MSE, CP, AL, MSE_pcr = simulate(n, iteration)
        push!(results[n]["MSE"], MSE)
        push!(results[n]["CP"], CP)
        push!(results[n]["AL"], AL)
        push!(results[n]["MSE_pcr"], MSE_pcr)
    end
end
