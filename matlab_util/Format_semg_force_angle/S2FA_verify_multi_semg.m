clear; close all;

semg_channel_index = 1:2;

force_channel_index = 1;
angle_channel_index = 2;

test_file_location = '../../LSTM/data/output/';
test_file_name = 'res_S2FA_DS200_SEG_2.txt';

train_file_location = '../../LSTM/data/input/';
train_file_name = 'exp_S2FA_DS200_SEG_2.txt';


[num_matrix, test_input_matrix_list, test_output_matrix_list] = ...
    read_test_file(strcat(test_file_location, test_file_name));

[num_matrix, train_input_matrix_list, train_output_matrix_list] = ...
    read_test_file(strcat(train_file_location, train_file_name));

%%
RMS_list = zeros(num_matrix, 1);
guess_RMS_list = zeros(num_matrix, 1);
for i = 1 : 10 %num_matrix
    test_semg_data = test_input_matrix_list{i}(:, semg_channel_index);
    test_force_data = test_output_matrix_list{i}(:, force_channel_index);
    test_angle_data = test_output_matrix_list{i}(:, angle_channel_index);
    
    DATA_LENGTH = length(test_output_matrix_list{i});
    
    train_semg_data = train_input_matrix_list{i}(:, semg_channel_index);
    train_force_data = train_output_matrix_list{i}(:, force_channel_index);
    train_angle_data = train_output_matrix_list{i}(:, angle_channel_index);
    
    figure;
    subplot_helper(1:DATA_LENGTH, test_semg_data, ...
                    [3 1 1], {'sample' 'amplitude' 'sEMG'}, ':x');
                
	subplot_helper(1:length(train_force_data), train_force_data, ...
                    [3 1 2], ...
                    {'sample' 'amplitude' 'force'}, ...
                    '-');                
    subplot_helper(1:length(test_force_data), test_force_data, ...
                    [3 1 2], ...
                    {'sample' 'amplitude' 'force'}, ...
                    '-'); 
                
	subplot_helper(1:length(train_angle_data), train_angle_data, ...
                    [3 1 3], ...
                    {'sample' 'amplitude' 'angle'}, ...
                    '-');                
    subplot_helper(1:length(test_angle_data), test_angle_data, ...
                    [3 1 3], ...
                    {'sample' 'amplitude' 'angle'}, ...
                    '-');                 
               
    legend('real', 'predict');
    ylim([-1 1]);
% 	print(strcat('./pics/', test_file_name, num2str(i), '.png'),'-dpng')
%    RMS_list(i) = sqrt(mean((train_force_data - test_force_data).^2));
%    guess_RMS_list(i) = sqrt(mean((train_force_data - 0.5*ones(size(train_force_data))).^2));
end

fprintf("mean RMS = %f\n", mean(RMS_list));
fprintf("guess mean RMS = %f\n", mean(guess_RMS_list));