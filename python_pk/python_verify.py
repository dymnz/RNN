import numpy as np
import operator


y_array = [[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 1, 0],
[0, 0, 0, 1],
[1, 0, 0, 0]]

x_array = [[1, 0, 0, 0],
		[0, 1, 0, 0],
		[0, 0, 1, 0],
		[0, 0, 1, 0],
		[0, 0, 0, 1]];

o_array = [[0.261640, 0.300283, 0.222404, 0.215673],
[0.281388, 0.340041, 0.129619, 0.248952],
[0.294358, 0.388210, 0.097885, 0.219547],
[0.294673, 0.392317, 0.095819, 0.217191],
[0.294643, 0.393357, 0.095500, 0.216499]]

s_array = [[0.000086, 0.419360, 0.301945, 0.045121],
[0.902741, 0.676738, 0.841098, 0.533531],
[0.989336, 0.969027, 0.976693, 0.957018],
[0.996255, 0.990323, 0.995270, 0.980730],
[0.993351, 0.992722, 0.997994, 0.985414]]

u_array = [[0.00009, 0.44692, 0.31166, 0.04515],
[0.86044, 0.40855, 0.51749, 0.46751],
[0.47226, 0.28162, 0.21852, 0.71358],
[0.14062, 0.38029, 0.58808, 0.82670]]

v_array = [[0.38264, 0.08468, 0.21757, 0.73526],
[0.59235, 0.55450, 0.47874, 0.23480],
[0.30784, 0.79622, 0.01161, 0.19311],
[0.57599, 0.71310, 0.01353, 0.38418]]

w_array = [[0.88698, 0.52211, 0.09690, 0.61058],
[0.95902, 0.32052, 0.90141, 0.01873],
[0.71743, 0.80805, 0.94926, 0.28691],
[0.16670, 0.79954, 0.94798, 0.73178]]

def softmax(x):
	xt = np.exp(x - np.max(x))
	return xt / np.sum(xt)

def bptt(y, s, o):
    T = len(y)
    # We accumulate the gradients in these variables    
    dLdV = np.zeros((4, 4), np.float32)
    #delta_o = o
    delta_o = np.subtract(o, y)
    # For each output backwards...
    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)       
    return [dLdV,dLdV]



def bptt_UVW(x, y, s, o, U, V, W, bptt_truncate = 4):
    T = len(y)
    # We accumulate the gradients in these variables
    dLdU = np.zeros(U.shape)
    dLdV = np.zeros(V.shape)
    dLdW = np.zeros(W.shape)
    delta_o = np.subtract(o, y)
    # For each output backwards...

    for t in np.arange(T)[::-1]:
        dLdV += np.outer(delta_o[t], s[t].T)
        # Initial delta calculation
        delta_t = V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
        print('for t = {}'.format(t))
        print('------------delta_t')
        print(delta_t)
        # Backpropagation through time (for at most self.bptt_truncate steps)
        for bptt_step in np.arange(max(0, t-bptt_truncate), t+1)[::-1]:
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            dLdW += np.outer(delta_t, s[bptt_step-1])          
            # print(dLdU[:,x[bptt_step]])            
            # dLdU[:,x[bptt_step]] += delta_t
            for r in np.arange(len(x[bptt_step])):
                if x[bptt_step][r] == 1:
                    dLdU[:, r] += delta_t 

            # Update delta for next step
            kkk = W.T.dot(delta_t)
            delta_t = W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)

            print('for t = {} bptt = {}'.format(t, bptt_step))
            print('------------W.T.dot(delta_t)')
            print(kkk)
            print('------------s[{}]'.format(bptt_step-1))
            print((1 - s[bptt_step-1] ** 2))
            print('------------delta_t')
            print(delta_t)
            # print('------------dLdU')
            # print(dLdU.T)
            # print('------------dLdV')
            # print(dLdV.T)
            # print('------------dLdW')
            # print(dLdW.T)
            #exit()

        #exit()
        
    return [dLdU, dLdV, dLdW]


def forward_propagation(x, U, V, W):
    # The total number of time steps
    T = len(x)
    # During forward propagation we save all hidden states in s because need them later.
    # We add one additional element for the initial hidden, which we set to 0
    s = np.zeros((T + 1, 4))
    s[-1] = np.zeros(4)
    # The outputs at each time step. Again, we save them for later.
    o = np.zeros((T, 4))
    # For each time step...
    for t in np.arange(T):
        # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
        s[t] = np.tanh(U.dot(x[t]) + W.dot(s[t-1]))
        o[t] = softmax(V.dot(s[t]))
    return [o, s]

def calculate_total_loss(x, y, U, V, W):
    L = 0
    y = y[0]
    # For each sentence...
    o, s = forward_propagation(x[0], U, V, W)
    print('------- o/s')
    print(o)
    print(s)
    for i in np.arange(len(y)):       
        # We only care about our prediction of the "correct" words
        for r in np.arange(len(y[i])):
            if y[i][r] == 1:
                correct_word_predictions = o[i][r]

        # Add to the loss based on how off we were
        #print(correct_word_predictions)
        print('L at {}'.format(i))        
        print(correct_word_predictions)
        L += -1 * np.sum(np.log(correct_word_predictions))
        print(L)
    return L

def calculate_loss(x, y, U, V, W):
    # Divide the total loss by the number of training examples
    N = np.sum((len(y_i) for y_i in y))
    print("NNNNNNNNNNNNNNNNNNNNNNNNn {}".format(N))
    return calculate_total_loss(x,y)/N

def gradient_check(x, y, s, o, U, V, W, h=0.001, error_threshold=0.01):
    # Calculate the gradients using backpropagation. We want to checker if these are correct.
    #bptt_gradients = bptt(y, s, o)
    bptt_gradients = bptt_UVW(x, y, s, o, U, V, W, 4)
    # List of all parameters we want to check.
    model_parameters = ['W']
    # Gradient check for each parameter
    pidx = 2
    pname = 'W'
    parameter = W
    print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
    # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
    it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
       
        # Save the original value so we can reset it later
        original_value = parameter[ix]
        # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
        print('=============================total_loss_mplus');
        parameter[ix] = original_value + h
        print('-====UVW')        
        print(U)        
        gradplus = calculate_total_loss([x],[y], U, V, W)
        print('=============================total_loss_minus');
        parameter[ix] = original_value - h
        print('-=====UVW')        
        print(U)
        gradminus = calculate_total_loss([x],[y], U, V, W)
        estimated_gradient = (gradplus - gradminus)/(2*h)
        # Reset parameter to original value
        parameter[ix] = original_value
        # The gradient for this parameter calculated using backpropagation
        backprop_gradient = bptt_gradients[pidx][ix]
        # calculate The relative error: (|x - y|/(|x| + |y|))
        relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
        # If the error is to large fail the gradient check
        if  relative_error > error_threshold: 
        #if ix == (0, 0):
            print "Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix)
            print "+h Loss: %f" % gradplus
            print "-h Loss: %f" % gradminus
            print "Estimated_gradient: %f" % estimated_gradient
            print "Backpropagation gradient: %f" % backprop_gradient
            print "Relative Error: %f" % relative_error
            return
        it.iternext()
    print "Gradient check for parameter %s passed." % (pname)


s = np.array(s_array)
o = np.array(o_array)
y = np.array(y_array)
x = np.array(x_array)

U = np.array(u_array).T
V = np.array(v_array).T
W = np.array(w_array).T

[p_o, p_s] = forward_propagation(x, U, V, W)

# print('------- o & p_o')
# print(o)
# print(p_o)

# print('------- s & p_s')
# print(s)
# print(p_s)
# exit()
# dLdV = bptt(y, s, o)
# print('------------dLdV')
# print(dLdV[0])

[dLdU, dLdV, dLdW] = bptt_UVW(x, y, p_s, o, U, V, W, 4)
print('------------U')
print(U.T)
print('------------V')
print(V.T)
print('------------W')
print(W.T)
print('------------dLdU')
print(dLdU.T)
print('------------dLdV')
print(dLdV.T)
print('------------dLdW')
print(dLdW.T)
exit()
gradient_check(x, y, p_s, o, U, V, W)

#dLdV = bptt(y, s, o)
#print(dLdV.T)
