using LinearAlgebra, Distributions, Random, SparseArrays, Plots, Statistics, PyCall
# using Conda
# Conda.add("scikit-learn")
roc_auc_score = pyimport("sklearn.metrics")["roc_auc_score"] # import sklearn.metrics.roc_auc_score 

include("Functions/logistic_fun.jl")
using .Logistic_function # module

include("Functions/pcr_fun.jl")
using .PCR_function # module

# function for numerical simulation
function simulate(n, iteration)
    rate = 4/3
    p = round(Int, n^rate)
    D = 10 * exp.(- (1:p) / 7) .+ n * exp(-n^(1/3)) / p
    U = qr(randn(p, p)).Q  # an orthonormal matrix
    Sigma = Diagonal(D)
    mvn = MvNormal(zeros(p), Sigma)
    theta = randn(p) 
    X = (U *  hcat([rand(mvn) for _ in 1:n]...))'
    y = [rand(Bernoulli(mu)) for mu in sigmoid.(X * theta)]
    
    # perform estimation
    mu, CholCov = VI_logistic(y, X, n*60)  
    CholCov = Diagonal(CholCov)

    max_components = min(size(X, 2), 30)  # set the upper limit to 30
    best_num_components = tune_num_components_logistic(X, y, max_components)

    # generate test data
    X_test = (U *  hcat([rand(mvn) for _ in 1:30000]...))'
    y_test = [rand(Bernoulli(mu)) for mu in sigmoid.(X_test * theta)]

    # prediction 
    posterior_sample = hcat([rand(MvNormal(mu, CholCov*CholCov')) for _ in 1:5000]...)
    posterior_prediction = sigmoid.(X_test * posterior_sample)
    y_pred_probs = mapslices(row -> mean(row), posterior_prediction; dims=2)
    y_pred = Int.(y_pred_probs .>= 0.5)
    posterior_upper = mapslices(row -> quantile(row, 0.975), posterior_prediction; dims=2)
    posterior_lower = mapslices(row -> quantile(row, 0.025), posterior_prediction; dims=2)
    # Proportion of misclassifications that were uncertain
    RA = sum((abs.(y_test - y_pred) .> 0) .* (posterior_upper .> 0.5 .> posterior_lower)) / sum(abs.(y_test - y_pred) .> 0)
    # Proportion of confident predictions that were correct
    RB = sum((abs.(y_test - y_pred) .== 0) .* .!(posterior_upper .> 0.5 .> posterior_lower)) / sum(.!(posterior_upper .> 0.5 .> posterior_lower))

    # prediction via pcr
    y_pred_pcr = PCR_logistic_predict(X, y, X_test, best_num_components)
    AUC_pcr = roc_auc_score(y_test, y_pred_pcr)
    MAE_pcr = mean(abs.(y_test .- Int.(y_pred_pcr .>= 0.5)))

    # calculate MAE and AUC (cross-entropy is not good because it diverges)
    AUC = roc_auc_score(y_test, y_pred_probs) 
    MAE = mean(abs.(y_test .- y_pred))  # 0-1 loss
    return MAE, AUC, RA, RB, MAE_pcr, AUC_pcr
end

results = Dict()
for n in 50:50:500
    results[n] = Dict("MAE" => [], "AUC" => [], "RA" => [],"RB" => [], "MAE_pcr" => [], "AUC_pcr" => [])
    for iteration in 1:20
        MAE, AUC, RA, RB, MAE_pcr, AUC_pcr = simulate(n, iteration)
        push!(results[n]["MAE"], MAE)
        push!(results[n]["AUC"], AUC)
        push!(results[n]["RA"], RA)
        push!(results[n]["RB"], RB)
        push!(results[n]["MAE_pcr"], MAE_pcr)
        push!(results[n]["AUC_pcr"], AUC_pcr)
    end
end
