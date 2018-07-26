
%% Initialize parameters
min_sampling_rate = 0.25;
sampling_rate = 0.2;

%% Create s sparse x
n = 3000;
s = floor(sqrt(n*min_sampling_rate/log(n)));
s=20;

% make x: s-sparse
x = zeros(n,1);
indices = randperm(length(x));
indices = indices(1:s);
x(indices,1) = randn(s,1);


%% Create Measurements
size_x = size(x);
m = round(n*sampling_rate);
mu = 0; sigma = 1;
shape = [m,n];
A = normrnd(mu,sigma,shape);
y = abs(A*x);


%% Phase Retrieval
tol1 = 0.001; tol2 = 0.001;
maxiter = 200;
[x_hat,err_hist,p,x_init] = CoPRAM(y,A,s,maxiter,tol1,tol2,x);


%% Print and save results
image = reshapeVintosquareM(x);
image_hat = reshapeVintosquareM(x_hat);

figure;
subplot(1,2,1);
imagesc(image); colormap gray; axis off;
title('Original Image');
subplot(1,2,2);
imagesc(image_hat); colormap gray; axis off;
title('Reconstructed Image');



%% utils

function M = reshapeVintosquareM (V)
% Reshapes a vector into the smallers square matrix possible. If V is too short
% the remainder of M is filled with NaNs.
N = numel(V) ;
M = NaN(ceil(sqrt(N))) ;
M(1:N) = V ;
end
