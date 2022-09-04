% load EIS vs SOC data
close all
clear
clc

flag_normalization = 0

% populate the list data acquisitions to be loaded
[battery_code,prefix] = get_data_path();

% --------------------------------------------
% Exported CSV file prefix
%  
file_prefix ='../results/FIT_MES'+battery_code+'_';
% --------------------------

SOC_vector = 100 : -10 : 10;

for idx_batt = 1 : length(prefix)
    
    disp(prefix(idx_batt))
    
    for idx_SOC = 1 : length(SOC_vector)
        load([prefix{idx_batt} 'SOC_' num2str(SOC_vector(idx_SOC)) '.mat']);
        disp(['OCV = ', num2str(mean_sig_v+4.12)]);
        
        % normalization
        if flag_normalization == 1
            Z_vector = (Z_vector - real(Z_vector(end)));
            Z_vector = Z_vector/max(real(Z_vector));
        end
        
        Z_matrix(:,idx_SOC,idx_batt) = Z_vector(:);

        % equivalent circuit fitting
        [Z_vector_fitted, x_hat] = func_fit_model_3(Z_vector, f0_vector);
        Z_matrix_fitted(:,idx_SOC,idx_batt) = Z_vector_fitted(:);
        x_hat_matrix(:,idx_SOC,idx_batt) = x_hat(:);
        
    end

    batt_matrix=x_hat_matrix(:,:,idx_batt);                   
    filename=strcat(file_prefix,string(idx_batt),'_ALL_SOC.csv');
    writematrix(batt_matrix,filename);         
end

