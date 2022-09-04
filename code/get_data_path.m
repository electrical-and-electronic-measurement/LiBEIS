% Use this file to select which data acquisition files will be loaded
% 1) You must choose one battery and set the "battery_code" variable
% 2) Fill the "data_path_list" with all folders containing data acquisition files
% BE CAREFULL:  Select only data acquisition files from the selected battery.
%
function [battery_code, data_path_list]= get_data_path()
    battery_code="03";
    data_path_list = {...
        '/data/Batt3/Meas4/';
        '/data/Batt3/Meas5/';
        '/data/Batt3/Meas6/';
        '/data/Batt3/Meas7/';
        '/data/Batt3/Meas8/';
        '/data/Batt3/Meas9/';
        };
end
