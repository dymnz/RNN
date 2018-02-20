function sig = dtanh(value)
sig = 1 - tanh(value) .* tanh(value);