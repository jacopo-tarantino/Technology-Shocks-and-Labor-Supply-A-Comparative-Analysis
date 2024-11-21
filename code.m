clear;


% 0. import data

fred_ohp = readmatrix('fred_ohp.csv');
data_o = fred_ohp(5:(end - 1), 2);
data_h = fred_ohp(5:(end - 1), 3); 
data_p = fred_ohp(5:(end - 1), 4);
% labor productivity
y = log(data_o ./ data_h);
% per-capita hours 
h = log(data_h ./ data_p);

% 1. estimate a VAR(4) for [∆yt ∆ht]'

data_tot = [diff(y) , diff(h)];
p = 4;
% plot time series
figure(1)
var = ["\Deltay" "\Deltah"];
plot(1:size(data_tot, 1), data_tot, ".-");
title("(\Deltay, \Deltah)");
xlabel("quarters");
legend(var, location = 'southwest');
% estimate VAR(4)
[par, Y, X] = estim_var(data_tot, p);

%% 2. derive the Wold representation (and plot) 

% number of variables
n = size(data_tot,2);   
% number of coefficients want to consider of VMA(inf)
K = 25;
% wold irf
F = [par(2:end, :)'; eye(n * (p - 1)) zeros(n * (p - 1), n)]; 	% step 1: create F matrix
for k=1:K
	CC = F^(k - 1);												% step 2: get the F^k for each K    
	C(:, :, k) = CC(1:n, 1:n);									% step 3: get only top-left of F^(k - 1)
end
% boorstrap
J = 1000;
[par, parBoot, sigBoot] = estim_var_boot(data_tot, p, J); 
for j = 1:J
	FBoot = [parBoot(2:end, :, j)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];	% step 1
	for k = 1:K
		CCBoot = FBoot^(k - 1);													% step 2
		CBoot(:, :, k, j) = CCBoot(1:n, 1:n);									% step 3
	end
end
% plot
figure(2)
var = ["\Deltay" "\Deltah"];
shock = ["\epsilon^{1}" "\epsilon^{2}"];
ii = 0;
for j = 1:n
	for i = 1:n
		ii = ii + 1;
		subplot(n, n, ii);
		arr = squeeze(C(j, i, :))';
		band = squeeze(prctile(CBoot(j, i, :, :), [16 84], 4))';
		hold on;
		fill([1:K, flip(1:K)], [band(1, :)'', flip(band(2, :))''], 'g', 'FaceAlpha', 0.3, 'LineStyle', '--');
		plot(1:K, band, "--k");
		plot(1:K, arr, ".-k");
		title("Impulse response of " + var(j) + " to the shock " + shock(i));
		xlabel("lags");	
		ylabel("estimated impulse response");
		hold off;
	end
end

% 3. identify the technology shock (and plot)

% structural irf
e = Y - (X * par);				% residuals
E = cov(e);						% covariance matrix
C1 = sum(C, 3);					% sum of C matrices
S = (chol(C1 * E * (C1')))';	% cholesky factor
A01 = inv(C1) * S;				% A_0^(-1)
for i = 1:K
    B(:, :, i) = C(:, :, i)  * A01;					% B(L) = C(L) x A_0^(-1)
    % B(:,:,i) = sum(C(:, :, 1:i), 3) * A01;        % B(L) = C(L) x A_0^(-1)
end
% bootstrap
J = 1000;
for j = 1:J
	% wold irf
	FBoot = [parBoot(2:end, :, j)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];	% step 1
	for k = 1:K
		CCBoot = FBoot^(k-1);													% step 2
		CBoot(:, :, k, j) = CCBoot(1:n, 1:n);									% step 3
	end
	% structural irf
	C1Boot(:, :, j) = sum(CBoot(:, :, :, j), 3);								% sum of C matrices
	SBoot = (chol(C1Boot(:, :, j) * sigBoot(:, :, j) * (C1Boot(:, :, j)')))';	% cholesky factor
	A01Boot = inv(C1Boot(:, :, j)) * SBoot;										% A_0^(-1)
	for i = 1:K
		BBoot(:, :, i, j) = CBoot(:, :, i, j) * A01Boot;						%B(L) = C(L) x Ao^(-1)
		% BBoot(:, :, i, j) = sum(CBoot(:, :, 1:i, j), 3) * A01Boot; 
	end
end
% plot
figure(3)
var = ["\Deltay" "\Deltah"];
shock = ["\eta^{1}" "\eta^{2}"];
ii = 0;
for j = 1:n
	for i = 1:n
		ii = ii + 1;
		subplot(n, n, ii);
		arr = squeeze(B(j, i, :))';
		band = squeeze(prctile(BBoot(j, i, :, :), [16 84], 4))';
		hold on;
		fill([1:K, flip(1:K)], [band(1, :)'', flip(band(2, :))''], 'g', 'FaceAlpha', 0.3, 'LineStyle', '--');
		plot(1:K, band, "--k");
		plot(1:K, arr, ".-k");
		title("Impulse response of " + var(j) + " to the shock " + shock(i));
		xlabel("lags");	
		ylabel("estimated impulse response");
		hold off;
	end
end
% plot (cumulated)
figure(4)
var = ["y" "h"];
shock = ["\eta^{1}" "\eta^{2}"];
ii = 0;
for j = 1:n
	for i = 1:n
		ii = ii + 1;
		subplot(n, n, ii);
		B_cum = cumsum(B, 3);
		BBoot_cum = cumsum(BBoot, 3);
		arr = squeeze(B_cum(j, i, :))';
		band = squeeze(prctile(BBoot_cum(j, i, :, :), [16 84], 4))';
		hold on;
		fill([1:K, flip(1:K)], [band(1, :)'', flip(band(2, :))''], 'g', 'FaceAlpha', 0.3, 'LineStyle', '--');
		plot(1:K, band, "--k");
		plot(1:K, arr, ".-k");
		title("Impulse response of " + var(j) + " to the shock " + shock(i));
		xlabel("lags");	
		ylabel("estimated impulse response");
		hold off;
	end
end

% 4. variance decomposition

V_tot = sum(sum(B.^2, 3), 2);			% tot variance of ∆yt ∆ht (sum for k and then for both variables) 
V_tecnology = sum(B(:, 1, :).^2, 3);	% variance of ∆yt ∆ht due to tecnology shock
V_decomposition = V_tecnology ./ V_tot * 100; 
display(V_decomposition);

% 5. estimate a VAR(4) for [∆yt ht]

data_tot_b = [diff(y) , h(2:end)];
p = 4;
[par_b, Y_b, X_b] = estim_var(data_tot_b, p); 

% 6. repeat step 3 with the new specification

n = size(data_tot_b, 2);				% number of variables
K;										% number of coefficients want to consider of VMA(inf) 
% wold irf
F_b = [par_b(2:end, :)'; eye(n * (p - 1)) zeros(n * (p - 1), n)]; 	% step 1: create F matrix
for k=1:K
	CC_b = F_b^(k - 1);												% step 2: get the F^k for each K    
	C_b(:, :, k) = CC_b(1:n, 1:n);									% step 3: get only top-left of F^(k - 1)
end
% structural irf
e_b = Y_b - (X_b * par_b);					% residuals
E_b = cov(e_b);								% covariance matrix
C1_b = sum(C_b, 3);							% sum of C matrices
S_b = (chol(C1_b * E_b * (C1_b')))';		% cholesky factor
A01_b = inv(C1_b) * S_b;					% A_0^(-1)
for i = 1:K
    B_b(:, :, i) = C_b(:, :, i)  * A01_b;	% B(L) = C(L) x A_0^(-1)
end
% bootstrap
J = 1000;
[par_b, parBoot_b, sigBoot_b] = estim_var_boot(data_tot_b, p, J); 
for j = 1:J
	% wold irf
	FBoot_b = [parBoot_b(2:end, :, j)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];		% step 1
	for k = 1:K
		CCBoot_b = FBoot_b^(k-1);														% step 2
		CBoot_b(:, :, k, j) = CCBoot_b(1:n, 1:n);										% step 3
	end
	% structural irf
	C1Boot_b(:, :, j) = sum(CBoot_b(:, :, :, j), 3);									% sum of C matrices
	SBoot_b = (chol(C1Boot_b(:, :, j) * sigBoot_b(:, :, j) * (C1Boot_b(:, :, j)')))';	% cholesky factor
	A01Boot_b = inv(C1Boot_b(:, :, j)) * SBoot_b;										% A_0^(-1)
	for i = 1:K
		BBoot_b(:, :, i, j) = CBoot_b(:, :, i, j) * A01Boot_b;							%B(L) = C(L) x Ao^(-1)
	end
end
% plot
figure(5)
ii = 0;
var = ["\Deltay" "h"];
shock = ["\eta^{1}" "\eta^{2}"];
j = 1;
for i = 1:n
	ii = ii + 1;
	subplot(n, n, ii);
	arr = squeeze(B_b(j, i, :))';
	band = squeeze(prctile(BBoot_b(j, i, :, :), [16 84], 4))';
	hold on;
	fill([1:K, flip(1:K)], [band(1, :)'', flip(band(2, :))''], 'g', 'FaceAlpha', 0.3, 'LineStyle', '--');
	plot(1:K, band, "--k");
	plot(1:K, arr, ".-k");
	title("Impulse response of " + var(j) + " to the shock " + shock(i));
	xlabel("lags");	
	ylabel("estimated impulse response");
	hold off;
end
% plot (cumulated)
figure(6)
var = ["y" "h"];
shock = ["\eta^{1}" "\eta^{2}"];
ii = 0;
for j = 1:n
	for i = 1:n
		ii = ii + 1;
		subplot(n, n, ii);
		B_b_cum = B_b;
		B_b_cum(1, :, :) = cumsum(B_b(1, :, :), 3);
		BBoot_b_cum = BBoot_b;
		BBoot_b_cum(1, :, :, :) = cumsum(BBoot_b(1, :, :, :), 3);
		arr = squeeze(B_b_cum(j, i, :))';
		band = squeeze(prctile(BBoot_b_cum(j, i, :, :), [16 84], 4))';
		hold on;
		fill([1:K, flip(1:K)], [band(1, :)'', flip(band(2, :))''], 'g', 'FaceAlpha', 0.3, 'LineStyle', '--');
		plot(1:K, band, "--k");
		plot(1:K, arr, ".-k");
		title("Impulse response of " + var(j) + " to the shock " + shock(i));
		xlabel("lags");	
		ylabel("estimated impulse response");
		hold off;
	end
end

% 7. repeat step 4 with the new specification

V_tot_b = sum(sum(B_b.^2, 3),2);   		% tot variance of ∆yt ht (sum for k and then for botth variables) 
V_tecnology_b = sum(B_b(:, 1, :).^2, 3);	% variance of ∆yt ht due to tecnology shock
V_decomposition_b = V_tecnology_b ./ V_tot_b * 100;
display(V_decomposition_b);

% functions

function[par, y, X, sig] = estim_var(data, p)
	% you lose the first p observations for each variable
	y = data(p + 1:end, :);     
	T = size(y, 1); 
	X = ones(T,1);  % (T - 1) x (p + 1)
	for j = 1:p
		X = [X data((p - j + 1):end - j, :)]; 
	end
	par = inv(X' * X) * X' * y;
	% covariance matrix 
	sig = cov(y - X * par);
end

function[par, parBoot, sig, F] = estim_var_boot(data, p, J)
	n = size(data, 2);
	[par, y, X] = estim_var(data, p);									%first estimation
	T = size(y, 1);														%size
	res = y - X * par;													% residuals
	% companion form for estimantion
	F = [par(2:end, :)'; eye(n * (p - 1)) zeros(n * (p - 1), n)];		%companion form coefficients
	C = [par(1, :)'; zeros(n * (p - 1), 1)];							%companion form costant
	% bootstrap
	z(:, 1) = X(1, 2:end)'; 											% z_1 non constructable, then equal to first observation
	for j = 1:J
		for t = 2:T
			E = [res(randi(T), :)'; zeros(n * (p - 1), 1)];				% randomly draw the residuals
			z(:, t) = C + F * z(:, t - 1) + E;							% data of previous step, new residual, F and C fixed
		end
		yBoot = [data(1:p, :); z(1:n, :)'];								% use old data for first p, then new data
		[parBoot(:, :, j), ~, ~, sig(:, :, j)] = estim_var(yBoot, p);	% estimate par on new data
	end
end	
