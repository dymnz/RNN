clear; close all;

file_name = '../../LSTM_one_hot/benchmark/batch.txt';
file = fileread(file_name);

% Parse
thread_num_strings = ...
    regexp(file, '(?<=Thread:[^0-9]*)[0-9]+', 'match');
hidden_num_strings = ...
    regexp(file, '(?<=Hidden:[^0-9]*)[0-9]+', 'match');
real_time_strings = ...
    regexp(file, '(?<=real[^0-9]*)[0-9 a-z .]+', 'match');
real_min_sec_strings = vertcat(...
    regexp(real_time_strings, '([^m]+)', 'match', 'once'), ...
    regexp(real_time_strings, '(?<=m)[0-9 .]+', 'match', 'once'));
epoch_num_strings = ...
    regexp(file, '(?<=Epoch:[^0-9]*)[0-9]+', 'match');

% To array
thread_num_array = str2double(thread_num_strings);
hidden_num_array = str2double(hidden_num_strings);
real_min_sec_array = str2double(real_min_sec_strings);
epoch_num_array = str2double(epoch_num_strings);

if (find(epoch_num_array~=10))
    disp('epoch error');
    return;
end

% 1 min = 60 sec
real_min_sec_array(2, :) = ...
    real_min_sec_array(1, :).*60 ...
    + real_min_sec_array(2, :);

% Merge [th, hid, sec]
data = horzcat(         ...
    thread_num_array',  ...
    hidden_num_array',  ... 
    real_min_sec_array(2, :)');

% hid-sec plot
figure;
for i = [1, 2, 4, 8, 16]
plot( ...
    data(thread_num_array==i, 2), ...
    data(thread_num_array==i, 3), ...
    '-o', 'LineWidth', 8);
hold on;
end

% Plot format
xlabel('hidden cell dim', 'FontSize', 20);
ylabel('time (sec)', 'FontSize', 20);
set(gca, 'FontSize', 20);
title('Time to 10 epochs @ hidden cell dim', 'FontSize', 30);
legend_list = {'1', '2', '4', '8', '16'};
legend(legend_list);


% hid-sec remove mean plot
figure;
for i = [1, 2, 4, 8, 16]
index = find(thread_num_array==i);
index(1)
offset = ...
    data(index(1), 3) - ...
    mean(data(thread_num_array==i, 3))
plot( ...
    data(thread_num_array==i, 2), ...
    data(thread_num_array==i, 3) - ...
    mean(data(thread_num_array==i, 3)) - offset,...
    '-o', 'LineWidth', 8);
hold on;
end

% Plot format
xlabel('hidden cell dim', 'FontSize', 20);
ylabel('time (sec)', 'FontSize', 20);
set(gca, 'FontSize', 20);
title('Time to 10 epochs @ hidden cell dim', 'FontSize', 30);
legend_list = {'1', '2', '4', '8', '16'};
legend(legend_list);
