% PROGRAM NAME: Aiyagari.m
clear, clc

% PARAMETERS
alpha = 0.33;
beta = .99; % discount factor
sigma = 1.5; % coefficient of risk aversion
delta=0.025;
rho = 0.5;
sigma_eps = 0.2;
m=5;
[z PI]=rouwenhorst(rho,sigma_eps,m); % endowment in employment states

PI_inv=PI^1000;
N=exp(z)* PI_inv(1,:)';% N=1

% ASSET VECTOR
a_lo = 0; %lower bound of grid points
a_hi = 100; %upper bound of grid points - can try upper bound of 3
num_a = 500; %try 700 points

a = linspace(a_lo, a_hi, num_a); % asset (row) vector

% INITIAL GUESS FOR K
K_min = 0;
K_max =500;
K_guess = (K_min + K_max) / 2;

% ITERATE OVER ASSET PRICES
K_tol = 1 ;
while abs(K_tol) >= 0.66 
    
    K_guess=(K_min+K_max)/2;
    r= alpha*K_guess^(alpha-1)*N^(1-alpha)+(1-delta); %rental rate
    w=(1-alpha)*K_guess^alpha*N^(-alpha); %wage rate
    

    % CURRENT RETURN (UTILITY) FUNCTION
    cons = bsxfun(@minus, r*a',  a); % where cons is 3 dim - a row vector, a', subtract q*a
    cons = bsxfun(@plus, cons, permute(z*w, [1 3 2])); % permute - rearranges - adding the third dimension
    ret = (cons .^ (1-sigma)) ./ (1 - sigma); % current period utility
    ret (cons<0) = -Inf;
   
    % INITIAL VALUE FUNCTION GUESS
    v_guess = zeros(m, num_a); % mxN
   
    % VALUE FUNCTION ITERATION
    v_tol = 1;
    while v_tol >.000001
   
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE
       
        value_mat = ret + beta * ...
            repmat(permute((PI * v_guess), [3 2 1]), [num_a 1 1]); % multiplying PI*v_guess - getting expectation
       
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
       
        [vfn, pol_indx] = max(value_mat, [], 2);
        vfn = permute(vfn,[3 1 2]);
       
        v_tol = abs(max(v_guess(:) - vfn(:)));
       
        v_guess = vfn; % update value functions
       
 
    end;
   
    % KEEP DECSISION RULE
    pol_indx=permute(pol_indx,[3 1 2]);
    pol_fn = a(pol_indx);
   
    % SET UP INITITAL DISTRIBUTION
    Mu=zeros(m,num_a); %any initial distribution works, as long as they sum up to 1 - can be uniform dist or put all mass at one point
    Mu(1,4) = 1; %suppose full mass at one point
   
    %function distribution = (pol_fn, PI);
   
    % ITERATE OVER DISTRIBUTIONS
   
    mu_tol = 1;
   
    while mu_tol> 0.00001
        [z_ind, a_ind, mass] = find(Mu > 0); % only looping over nonzero indices- find non-zero indices - employment and asset index
       
                 
        MuNew = zeros(size(Mu));
        for ii = 1:length(z_ind)
            apr_ind = pol_indx(z_ind(ii), a_ind(ii)); % which a prime does the policy fn prescribe?
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
                (PI(z_ind(ii), :) * Mu(z_ind(ii), a_ind(ii)))';
        end
   
        mu_tol = max(abs(MuNew(:) - Mu(:)));

        Mu = MuNew;

    end
   
   
    % AGGREGATE/ INTEGRATE AND CHECK FOR MARKET CLEARING
   
     % multiply MU * pol-fn, tells us how much total saving of people at any given state
   aggK= sum(sum( pol_fn(:) .* Mu(:) )); 
   K_tol=aggK-K_guess;
    
    if K_tol> 0.1
       K_min = K_guess;
    else
       K_max = K_guess;

    end
            
   
       
end
plot(a,pol_fn);
title('Aiyagari Policy function');

