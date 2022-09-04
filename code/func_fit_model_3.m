function [Z_hat, x_hat] = func_fit_model_3(Z_vector, f0_vector)

    re=real(Z_vector);
    im=imag(Z_vector);
    s=1i*2*pi*f0_vector';
    omega = 2*pi*f0_vector';


    %% initial guess
    R0 = real(Z_vector(end)); % high-frequency intercept with x axis
    R1 = (real(Z_vector(end)) - real(Z_vector(1))); % diameter of semicircle
    Q1 = 1/(2*pi*200*R1); % top of semicircle occurs at 200 Hz
    p1 = 0.5; % exponent
    Q2 = Q1*2; 
    p2 = 0.5; % exponent
    L = 1e-14;
        
    x0 = [R0 R1 Q1 p1 Q2 p2 L];


    %% bounds
    %   R0      R1      Q1      p1      Q2      p2      L
    xl=[0       0       0       0       0       0       0];
    xu=[0.1     0.1     1e3     1       1e3     1       1];


    %% optimization
    Z_CPE = @(x) 1./(x(1)*(1j*omega).^x(2));
    Z_func = @(x) x(1) + 1j*omega*x(7) ...
        + (Z_CPE([x(5) x(6)]) .* (x(2)+Z_CPE([x(3) x(4)]))) ...
        ./ (Z_CPE([x(5) x(6)]) + (x(2)+Z_CPE([x(3) x(4)])));

    fun = @(x)norm([...
        re - real(Z_func(x));
        im - imag(Z_func(x))...
        ]);
    
    tic
    problem = createOptimProblem('fmincon','objective',...
        fun, 'x0', x0, 'lb', xl, 'ub', xu);
    ms = MultiStart('UseParallel',true,'Display','off');
    [x_hat,f,EXITFLAG,OUTPUT,SOLUTIONS] = run(ms,problem,1e2);
    toc
    
%     x_hat 
    
    Z_hat = Z_func(x_hat);

    
end


