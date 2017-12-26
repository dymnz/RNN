function sig = dsigmoid(value)
sig = sigmoid(value) .* (1 - sigmoid(value));