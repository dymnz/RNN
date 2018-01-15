clear; close all;


test_file_location = '../../LSTM/data/output/';
test_file_name = 'res_2_first_half_stream_rect.txt';

train_file_location = '../../LSTM/data/input/';
train_file_name = 'exp_2_first_half_stream_rect.txt';

% test_file_location = '../../LSTM/data/output/';
% test_file_name = 'res_test_2_ds100_lp_rec_fx_ol.txt';
% 
% train_file_location = '../../LSTM/data/input/';
% train_file_name = 'exp_test_2_ds10s0_lp_rec_fx_ol.txt';

[num_matrix, test_input_matrix_list, test_output_matrix_list] = ...
    read_test_file(strcat(test_file_location, test_file_name));

[num_matrix, train_input_matrix_list, train_output_matrix_list] = ...
    read_test_file(strcat(train_file_location, train_file_name));

%%
for i = 1 : num_matrix
    test_semg_data = test_input_matrix_list{i};
    test_force_data = test_output_matrix_list{i};
    DATA_LENGTH = length(test_input_matrix_list{i});
    
    train_semg_data = train_input_matrix_list{i};
    train_force_data = train_output_matrix_list{i};

    figure;
    subplot_helper(1:DATA_LENGTH, test_semg_data, ...
                    [2 1 1], {'sample' 'amplitude' 'pulse'}, ':x');
    subplot_helper(1:length(test_force_data), test_force_data, ...
                    [2 1 2], ...
                    {'sample' 'amplitude' 'twitch force prototype'}, ...
                    '-');
    subplot_helper(1:length(train_force_data), train_force_data, ...
                    [2 1 2], ...
                    {'sample' 'amplitude' 'twitch force prototype'}, ...
                    '-');                
    ylim([0 1]);
   
end
