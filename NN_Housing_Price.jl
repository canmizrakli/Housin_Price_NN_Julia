using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean
using Parameters
using Distributions
using Random
using Flux
using MLUtils
using NNlib  
using Random
using PyCall 
using Dates
using Statistics
using Plots

# Load and preprocess the data
data = CSV.read("kc_house_data.csv", DataFrame)
select!(data, Not([:id, :date]))

# Shuffle and split the data
Random.seed!(1234)
n = size(data, 1)
train_size = Int(round(0.8 * n))
test_size = n - train_size
data = data[shuffle(1:n), :]

# split
train_data = data[1:train_size, :]
test_data = data[train_size+1:end, :]

# separate features and target
X_train = Matrix(train_data[:, Not(:price)])
y_train = convert(Vector, train_data[:, :price])
X_test = Matrix(test_data[:, Not(:price)])
y_test = convert(Vector, test_data[:, :price])

# normalize
X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
X_test = (X_test .- mean(X_test, dims=1)) ./ std(X_test, dims=1)
y_train = (y_train .- mean(y_train)) ./ std(y_train)
y_test = (y_test .- mean(y_test)) ./ std(y_test)

# create a model
function create_model(layer_sizes, learning_rate)
    model = Chain(
        Dense(layer_sizes[1], layer_sizes[2], relu),
        Dense(layer_sizes[2], layer_sizes[3], relu),
        Dense(layer_sizes[3], 1)
    )
    optimizer = ADAM(learning_rate)
    return model, optimizer
end

# model settings
models = [
    (layers = [size(X_train, 2), 128, 64], learning_rate = 0.01),
    (layers = [size(X_train, 2), 64, 32], learning_rate = 0.001),
    (layers = [size(X_train, 2), 128, 64], learning_rate = 0.0005)
]

history_list = []

# transpose data
X_train = transpose(X_train)
X_test = transpose(X_test)

# reshape data
y_train = reshape(y_train, 1, :)
y_test = reshape(y_test, 1, :)

# start the training loop
for config in models
    model, optimizer = create_model(config.layers, config.learning_rate)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    ps = Flux.params(model)
    
    for epoch in 1:100
        Flux.train!(loss, ps, [(X_train, y_train)], optimizer)
    end
    
    train_loss = loss(X_train, y_train)
    val_loss = loss(X_test, y_test)
    push!(history_list, (model = config, train_loss = train_loss, val_loss = val_loss))
end

for (i, history) in enumerate(history_list)
    println("Model ", i, " Learning Rate: ", history.model.learning_rate, " Train Loss: ", history.train_loss, " Validation Loss: ", history.val_loss)
end

# Initialize a list to store the loss for each epoch
epoch_losses = []

# Now start the training loop
for config in models
    model, optimizer = create_model(config.layers, config.learning_rate)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    ps = Flux.params(model)
    
    for epoch in 1:100
        Flux.train!(loss, ps, [(X_train, y_train)], optimizer)
        
        # Compute the loss for this epoch and store it
        train_loss = loss(X_train, y_train)
        val_loss = loss(X_test, y_test)
        push!(epoch_losses, (model = config, epoch = epoch, train_loss = train_loss, val_loss = val_loss))
    end
end

# extract the unique models
models = unique([x.model for x in epoch_losses])

p = plot()  # initialize an empty plot

for model in models
    
    model_losses = filter(x -> x.model == model, epoch_losses)
    local train_losses = [x.train_loss for x in model_losses]
    local val_losses = [x.val_loss for x in model_losses]
    plot!(p, train_losses, label = "Training loss - Model $(model)")
    plot!(p, val_losses, label = "Validation loss - Model $(model)")

end

xlabel!(p, "Epochs")
ylabel!(p, "Loss")

p  # display plot