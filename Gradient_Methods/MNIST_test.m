
%% Initialize parameters
sampling_rate = 0.5;

%% Load image
dir = "data/MNIST/";
filename = "mnist2.mat";
my_struct = load(dir+filename);
array = getfield(my_struct,'vect');
newsize = [28, 28];
image = transpose(reshape(array, newsize));

%% Sparsify / Wavelet Transform 
img_size = size(image);
wname = 'bior3.5';
level = 5;
[x_transform,S] = wavedec2(image,level,wname);
sparsity = get_approx_sparsity(x_transform, 0.00001);


%% Create Measurements

% % For MNIST directly
% [numrows,numcols] = size(image);
% n = numrows*numcols;
% m = round(n*sampling_rate);
% mu = 0; sigma = 1;
% shape = [m,n];
% A = normrnd(mu,sigma,shape);
% 
% x = image(:); 
% b = abs(A*x);

% For Sparsified output
x_transform = transpose(x_transform);
size_xtrans = size(x_transform);
n = size_xtrans(1);
m = round(n*sampling_rate);
mu = 0; sigma = 1;
shape = [m,n];
A = normrnd(mu,sigma,shape);

x = image(:); 
y = abs(A*x_transform);


%% Phase Retrieval
tol1 = 0.01; tol2 = 0.01;
maxiter = 100;
[x_transform_hat,err_hist,p,x_init] = CoPRAM(y,A,sparsity,maxiter,tol1,tol2,x_transform);

%% Inverse Sparse Transfrom
x_transform_hat = transpose(x_transform_hat);
thr = wthrmngr('dw2ddenoLVL','penalhi',x_transform_hat,S,3);
sorh = 's';
[image_hat,cfsDEN,dimCFS] = wdencmp('lvd',x_transform_hat,S,wname,level,thr,sorh);


%% Print and save results

figure;
subplot(1,2,1);
imagesc(image); colormap gray; axis off;
title('Original Image');
subplot(1,2,2);
imagesc(image_hat); colormap gray; axis off;
title('Reconstructed Image');



%% utils

function s = get_approx_sparsity(array,threshold)
arr_sz = size(array);
s = 0;

for i= 1:arr_sz(2)
    if(array(i) > threshold)
        s = s +1;
    end
end

end
