DATA_LENGTH = 100;
RAND_THRESHOLD = 1;

% Twitch force param %
P = 1;     % Peak force
CT = 5;    % Time to peak


RAND_THRESHOLD = rand * 3 - 0.01;
[pulse_vector, tf_vector, tf_proto] = ...
                random_twitch_force(DATA_LENGTH, RAND_THRESHOLD, P, CT);
sum(tf_vector)  

CT = 2;
[pulse_vector, tf_vector, tf_proto] = ...
                random_twitch_force(DATA_LENGTH, RAND_THRESHOLD, P, CT);         
            
sum(tf_vector)              