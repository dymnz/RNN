clear; close all;

% Twitch force param %
P = 1;     % Peak force
CT = 5;    % Time to peak

file_location = '../C/test_data/';
%file_name = 'demo/res10_CT2.txt';
%file_name = 'demo/res2.txt';
file_name = 'demo/res10_t2.txt';
%file_name = 'exp2_1_CT5.txt';
%% 

[num_matrix, input_matrix_list, output_matrix_list] = ...
    read_test_file(strcat(file_location, file_name));

RMSE_list = zeros(num_matrix, 1);

for i = 1 : num_matrix
    pulse_vector = input_matrix_list{i};
    predicted_tf_vector = output_matrix_list{i};
    DATA_LENGTH = length(input_matrix_list{i});
    
	[expected_tf_vector, tf_proto] = ...
        twitch_force_vector_gen(pulse_vector, P, CT, DATA_LENGTH);

    delta_matrix = predicted_tf_vector - expected_tf_vector;
    
    RMSE_list(i) = sqrt(sum(delta_matrix.^2)/DATA_LENGTH);
    fprintf('sample %d loss: %f\n', i, RMSE_list(i));
    
    figure;
    subplot_helper(1:DATA_LENGTH, pulse_vector, ...
                    [4 1 1], {'sample' 'amplitude' 'pulse'}, ':x');
    subplot_helper(1:length(tf_proto), tf_proto, ...
                    [4 1 2], ...
                    {'sample' 'amplitude' 'twitch force prototype'}, ...
                    '-');
    ylim([0 1]);
    subplot_helper(1:DATA_LENGTH, pulse_vector, ...
                    [4 1 3], {'sample' 'amplitude' 'force'}, ':x');   
    subplot_helper(1:DATA_LENGTH, expected_tf_vector, ...
                    [4 1 3], {'sample' 'amplitude' 'force'}, '-');    
    subplot_helper(1:DATA_LENGTH, predicted_tf_vector, ...
                    [4 1 3], {'sample' 'amplitude' 'force'}, '-'); 
    legend('pulse', 'expected', 'predicted');
    subplot_helper(1:DATA_LENGTH, abs(delta_matrix), ...
                    [4 1 4], {'sample' 'amplitude' '|force error|'}, '-'); 
    ylim([0 1]);
   
end

fprintf('average loss: %f\n', mean(RMSE_list));