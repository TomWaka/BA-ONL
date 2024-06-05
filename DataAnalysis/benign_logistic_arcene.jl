using LinearAlgebra, Distributions, Random, SparseArrays, Plots, Statistics, PyCall, CSV, DataFrames
# Import sklearn's ROC AUC score function via PyCall
roc_auc_score = pyimport("sklearn.metrics")["roc_auc_score"]

include("Functions/logistic_fun.jl")
using .Logistic_function # module

Random.seed!(123)

# Combine training and validation sets
combined_data = vcat(train_data, valid_data)
combined_labels = map(x -> x == -1 ? 0 : 1, Matrix(vcat(train_labels, valid_labels)))
n = size(combined_data, 1)
num_folds = 5

# Calculate fold size for cross-validation
fold_size = div(n, num_folds)

# Arrays to store performance metrics
AUC_scores = Float64[]
MAE_scores = Float64[]
RA_scores = Float64[]
RB_scores = Float64[]

for fold in 1:num_folds
    # Define indices for testing and training data
    test_indices = (fold - 1) * fold_size + 1 : min(fold * fold_size, n)
    train_indices = setdiff(1:n, test_indices)

    # Prepare training and testing sets
    X_train = Matrix(combined_data[train_indices, 1:10000])
    y_train = combined_labels[train_indices]
    X_test = Matrix(combined_data[test_indices, 1:10000])
    y_test = combined_labels[test_indices]

    # Train the model
    mu, CholCov = VI_logistic(y_train, X_train, 20000, 0.0003)
    CholCov = Diagonal(CholCov)

    # Evaluate the model
    posterior_sample = [rand(MvNormal(mu, CholCov * CholCov')) for _ in 1:10000]
    posterior_prediction = sigmoid.(X_test * hcat(posterior_sample...))
    y_pred_probs = mean(posterior_prediction, dims=2)
    y_pred = Int.(y_pred_probs .>= 0.5)
    y_sample = [rand(Bernoulli(mu)) for mu in posterior_prediction]
    posterior_upper = mapslices(row -> quantile(row, 0.975), y_sample; dims=2)
    posterior_lower = mapslices(row -> quantile(row, 0.025), y_sample; dims=2)
    # Proportion of misclassifications that were uncertain
    RA = sum((abs.(y_test - y_pred) .> 0) .* (posterior_upper .> 0.5 .> posterior_lower)) / sum(abs.(y_test - y_pred) .> 0)
    # Proportion of confident predictions that were correct
    RB = sum((abs.(y_test - y_pred) .== 0) .* .!(posterior_upper .> 0.5 .> posterior_lower)) / sum(.!(posterior_upper .> 0.5 .> posterior_lower))
    AUC = roc_auc_score(y_test, y_pred_probs)
    MAE = mean(abs.(y_test - y_pred))

    push!(AUC_scores, AUC)
    push!(MAE_scores, MAE)
    push!(RA_scores, RA)
    push!(RB_scores, RB)
end
