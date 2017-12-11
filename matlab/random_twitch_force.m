function [pulse_vector, tf_vector, tf_proto] = ...
            random_twitch_force(DATA_LENGTH, RAND_THRESHOLD, P, CT)

rand_vector = abs(normrnd(0, 1, DATA_LENGTH, 1));
pulse_vector = zeros(DATA_LENGTH, 1);

pulse_vector(rand_vector>RAND_THRESHOLD) = ...                                              ...
        rand_vector(1:length(find(rand_vector>RAND_THRESHOLD))) .*   ...
        rand_vector(rand_vector>RAND_THRESHOLD);

pulse_vector = pulse_vector ./ max(pulse_vector);    
    
% Generate twitch force vector
[tf_vector, tf_proto] = ...
    twitch_force_vector_gen(pulse_vector, P, CT, DATA_LENGTH);
