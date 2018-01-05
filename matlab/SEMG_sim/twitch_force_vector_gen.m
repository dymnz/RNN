function [tf_vector, tf_proto] = twitch_force_vector_gen(pulse_vector, P, CT, DATA_LENGTH)

tf_proto = twitch_force(1:CT*5, P, CT);

% Copy pasting
tf_vector = conv(pulse_vector, tf_proto);

% Truncation
tf_vector = tf_vector(1:DATA_LENGTH);

% Normalization
tf_vector = tf_vector ./ max(tf_vector);

