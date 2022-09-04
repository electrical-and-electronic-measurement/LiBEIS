% load EIS vs SOC data
close all
clear
clc

% populate the list data acquisitions to be loaded
[battery_code,prefix] = get_data_path();

% --------------------------------------------
% Exported CSV file prefix
%  
file_prefix ='../results/EIS_BATT'+battery_code+"_";
% --------------------------



SOC_vector = 100 : -10 : 10;

for idx_batt = 1 : length(prefix)
    
    disp(prefix(idx_batt))
    
    for idx_SOC = 1 : length(SOC_vector)
        load([prefix{idx_batt} 'SOC_' num2str(SOC_vector(idx_SOC)) '.mat']);
        disp(['OCV = ', num2str(mean_sig_v+4.12)]);

        % compute impedance from current (sig_i) and voltage (sig_v) time series 
        f_base = 0 : Fs/N : Fs - Fs/N;
        V = fft(sig_v)/N;
        I = (fft(sig_i)/N).*exp(-1j*2*pi*f_base(:)/(Fs*2.0)); % inter-channel delay correction
        idx_freq = round(N*f0_vector/Fs+1); % index of the bins of interest (excited frequencies)
        Z_vector = V(idx_freq)./I(idx_freq);
        
        Z_matrix(:,idx_SOC,idx_batt) = Z_vector(:);
                
        batt_matrix=Z_matrix(:,:,idx_batt);                   
        filename=strcat(file_prefix,string(idx_batt),'_ALL_SOC.csv');
        writematrix(batt_matrix,filename);         
        
        OCV(idx_SOC, idx_batt) = mean_sig_v+4.12;
        
    end
end


%% plot
figure(1);
for idx_SOC = 1 : length(SOC_vector)
    Z_temp = squeeze(Z_matrix(:,idx_SOC,:));
    
    figure(1);
    subplot(3,4,idx_SOC)
    plot(real(Z_temp), -imag(Z_temp));

    xlabel('Re(Z) [\Omega]')
    ylabel('-Im(Z) [\Omega]')
    axis([min(real(Z_matrix(:))) ...
            max(real(Z_matrix(:))) ... 
            min(-imag(Z_matrix(:))) ...
            max(-imag(Z_matrix(:)))]);
    
    title(['SOC ' num2str(SOC_vector(idx_SOC))]);
    grid on;
    hold on;
    
    Z_mean(:,idx_SOC) = mean(Z_temp,2);
%     plot(real(Z_mean(:,idx_SOC)), -imag(Z_mean(:,idx_SOC)),'k--');
%     grid on; hold on;
    
end

figure(1)
subplot(3,4,idx_SOC+1)
plot(real(Z_temp), -imag(Z_temp));
hold on;
% plot(real(Z_mean(:,idx_SOC)), -imag(Z_mean(:,idx_SOC)),'k--');
% legend([prefix; 'average'], 'Location', 'best')
legend(prefix, 'Location', 'best')
axis(1e3+[0 1 0 0.3])
set(gca,'visible','off')
saveas(gcf,'../results/EIS_curves.pdf')



% plot OCV
figure; 
plot(SOC_vector, OCV,'.-')
xlabel('SOC [%]')
ylabel('OCV [V]')
legend(prefix, 'Location', 'best')
saveas(gcf,'../results/OCV_vs_SOC.pdf')

figure;
plot(SOC_vector, OCV-mean(OCV,2),'.-')
xlabel('SOC [%]')
ylabel('OCV dispersion [V]')
legend(prefix, 'Location', 'best')
saveas(gcf,'../results/OCV_dispersion.pdf')



