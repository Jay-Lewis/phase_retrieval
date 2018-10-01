import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib as plt
import torch.optim
from src.models.skip import skip
from src.utils.inpainting_utils import *
from src.utils.utils import *


class DIP_net(object):

    def __init__(self, image, in_shape, image_name):
        self.dtype = torch.cuda.FloatTensor
        self.in_shape = in_shape
        self.x = image
        self.channels, self.n1, self.n2 = image.shape
        self.n = self.n1*self.n2
        self.x_flat = np.reshape(self.x, [1, self.n])
        self.net_input, self.net = self.build_network()
        self.loss = self.define_loss()
        self.iter = 0
        self.plot_dir = "../plots/DIP/"
        self.image_name = image_name

    def build_network(self):
        INPUT = 'meshgrid'
        input_depth = 2  # depth of input noise
        output_depth = self.channels
        pad = 'reflection'  # 'zero'

        num1 = 5  # 5 TODO: describe what these numbers represent
        num2 = 3  # 3
        num3 = 128  # 128

        net = skip(input_depth, output_depth,
                   num_channels_down=[num3] * num1,
                   num_channels_up=[num3] * num1,
                   num_channels_skip=[0] * num1,
                   upsample_mode='nearest', filter_skip_size=1, filter_size_up=num2, filter_size_down=num2,
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(self.dtype)

        net = net.type(self.dtype)
        net_input = get_noise(input_depth, INPUT, self.in_shape).type(self.dtype)

        return net_input, net

    def define_loss(self):
        mse = torch.nn.MSELoss().type(self.dtype)
        return mse

    def reconstruct(self, A_type, reconstruct_mode, mn_ratio, num_iter, save_flag):
        self.iter = 0
        self.A = self.get_A(mn_ratio, A_type)
        self.measurement = self.get_measurement(A_type, reconstruct_mode)
        self.measurement_var, self.A_var, self.x_var, self.real_A_var, self.img_A_var = self.get_torch_vars(A_type)
        self.optimize_setup()
        p = get_params(self.OPT_OVER, self.net, self.net_input)
        optimize_function_params = (A_type, mn_ratio, reconstruct_mode, save_flag)
        losses, img_losses = optimize(self.OPTIMIZER, p, self.optimize_function,
                                      self.LR, num_iter, *optimize_function_params)
        if(save_flag):
            self.plot_loss_curves(losses, img_losses, A_type, mn_ratio)

        return losses, img_losses

    def get_A(self, mn_ratio, A_type):

        if A_type == "gaussian":
            self.m = int(self.n * mn_ratio)
            mu = 0.0
            sigma = 1.0/self.m
            A = mu + sigma * np.random.randn(self.m, self.channels, self.n)

        elif A_type == "DFT":
            self.m = int(self.n * mn_ratio)
            A = np.fft.fft(np.eye(self.n))  # TODO: might need to divide by sqrt(n) *******
            A = A.reshape(self.n1, 1, self.n)

        elif A_type == "2d_DFT":
            self.m = int(self.n1 * mn_ratio)
            A = np.fft.fft(np.eye(self.n1))  # TODO: might need to divide by sqrt(n) *******
            A = A.reshape(self.n1, 1, self.n1)

        elif A_type == "oversampled_2d_DFT":
            self.m = int(self.n1 * mn_ratio)
            A = np.fft.fft(np.eye(self.m))  # TODO: might need to divide by sqrt(n) *******
            A = A[:, 0:self.n1]
            A = A.reshape(self.m, 1, self.n1)

        elif A_type == "c_gaussian":
            self.m = int(self.n * mn_ratio)
            mu = 0.0
            sigma = 1.0 / float(2 * self.m) # the variance of each element is half the variance of the real gaussian case
            A_real = mu + sigma * np.random.randn(self.m, self.channels, self.n)
            A_img = mu + sigma * np.random.randn(self.m, self.channels, self.n)
            A = A_real + 1j * A_img

        else:
            raise NotImplementedError('invalid A matrix qualifier')

        if self.channels != 1:
            raise ValueError('not yet implemented for images with more than one color channel')

        return A

    def get_measurement(self, A_type, reconstruct_mode):

        if(reconstruct_mode == 'linear'):
            if (A_type == 'gaussian'):
                y = np.tensordot(self.A, self.x_flat, 2)

            elif (A_type == 'c_gaussian' or A_type == 'DFT'):
                raise NotImplementedError

            else:
                raise NotImplementedError

        elif(reconstruct_mode == 'phase-retrieval'):
            if (A_type == 'gaussian' or A_type == 'c_gaussian' or A_type == 'DFT'):
                y = np.tensordot(self.A, self.x_flat, 2)
                y = np.abs(y)

            elif (A_type == '2d_DFT' or A_type == 'oversampled_2d_DFT'):    # Note: 2d dft modeled as M*X*M^T
                y_prime = np.tensordot(self.A, self.x, 2)
                A_t = np.transpose(np.reshape(self.A, [self.m, self.n1]))   # TODO: image must be symmetric (not checked)
                y = np.tensordot(y_prime, A_t, 1)
                y = np.abs(y)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return y

    def get_measurement_hat(self, out, A_type, reconstruct_mode):

        new_shape = (1, self.n)
        out_flat = out.view(new_shape)

        if (A_type == 'gaussian'):
            if (reconstruct_mode == 'linear'):
                measurement_hat = tensordot_pytorch(self.A_var, out_flat, axes=2)

            elif (reconstruct_mode == 'phase-retrieval'):
                measurement_hat = tensordot_pytorch(self.A_var, out_flat, axes=2)
                measurement_hat = measurement_hat.abs()
            else:
                raise NotImplementedError

        elif (A_type == 'c_gaussian' or A_type == 'DFT'):
            if (reconstruct_mode == 'linear'):
                raise NotImplementedError

            elif (reconstruct_mode == 'phase-retrieval'):
                measurement_hat = (tensordot_pytorch(self.real_A_var, out_flat, axes=2) ** 2
                                   + tensordot_pytorch(self.img_A_var, out_flat, axes=2) ** 2) ** (1 / 2)
            else:
                raise NotImplementedError

        elif (A_type == '2d_DFT' or A_type == 'oversampled_2d_DFT' ):   # 2d dft modeled as M*X*M^T
            if (reconstruct_mode == 'linear'):
                raise NotImplementedError

            elif (reconstruct_mode == 'phase-retrieval'):
                real_A_var_sliced = self.real_A_var.view(self.m, self.n1)
                img_A_var_sliced = self.img_A_var.view(self.m, self.n1)

                r_r_part = tensordot_pytorch(tensordot_pytorch(self.real_A_var, out, axes=2), real_A_var_sliced.transpose(0,1), axes =1)
                i_i_part = tensordot_pytorch(tensordot_pytorch(self.img_A_var, out, axes=2), img_A_var_sliced.transpose(0,1), axes =1)
                i_r_part = tensordot_pytorch(tensordot_pytorch(self.img_A_var, out, axes=2), real_A_var_sliced.transpose(0,1), axes =1)
                r_i_part = tensordot_pytorch(tensordot_pytorch(self.real_A_var, out, axes=2), img_A_var_sliced.transpose(0,1), axes =1)

                measurement_hat = ((r_r_part-i_i_part)**2 + (i_r_part+r_i_part)**2)**(1/2)    # TODO: image must be symmetric (not checked)

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return measurement_hat

    def sweep_m_comparison(self, mn_ratios, A_types, reconstruct_mode, num_iter):
        for A_type in A_types:
            self.sweep_m_plot(mn_ratios, A_type, reconstruct_mode,
                              num_iter, plot_flag=False, save_flag=False)
        plt.title('m sweep comparison')
        plt.legend()
        plt.savefig(self.plot_dir+"comparison/" + "m sweep comparison"+str(A_types)+".png")

    def sweep_m_plot(self, mn_ratios, A_type, reconstruct_mode, num_iter, plot_flag=True, save_flag=True):
        final_losses = []
        for mn_ratio in mn_ratios:
            _, img_losses = self.reconstruct(A_type, reconstruct_mode, mn_ratio, num_iter, save_flag)
            final_loss = img_losses[-1]
            final_losses.append(final_loss)

        xs = mn_ratios
        # plt.figure()
        plt.plot(xs, final_losses, label=str(A_type))
        plt.xlabel('m/n (sensing ratio)')
        plt.ylabel('mse (per pixel)')

        full_directory = self.get_full_directory(A_type)
        if(save_flag):  #TODO: NOT WORKING
            plt.title('m sweep ' + str(A_type))
            plt.savefig(full_directory+"m_sweep_"+str(A_type)+".png")

        if (plot_flag):
            plt.title('m sweep ' + str(A_type))
            plt.show()

        return

    def optimize_function(self, A_type, mn_ratio, reconstruct_mode, save_flag):

        # Add noise to network parameters / network input
        if self.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        self.net_input = self.net_input_saved
        if self.reg_noise_std > 0:
            self.net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)

        # Output of network
        out = self.net(self.net_input)
        out_np = torch_to_np(out)

        # Calculate measurement estimate |A*net_out|
        new_shape = out.shape[1:]  # eliminate unnecessary dimension
        out = out.view(new_shape)  # must reshape before tensordot
        measurement_hat = self.get_measurement_hat(out, A_type, reconstruct_mode)

        # Define Loss (||A*net_out|-|A*image||l2)
        total_loss = self.loss(measurement_hat, self.measurement_var)

        diff_squared = np.subtract(torch_to_np(measurement_hat), self.measurement)**2
        sum = np.sum(diff_squared)
        mse = sum/(np.prod(diff_squared.shape))
        # print(np.shape(total_loss))
        # print(total_loss)
        # print(mse)
        total_loss.backward()

        # Record Image Loss (considering x and -x as trivial ambiguities)
        total_loss_img = np.min([self.loss(out, self.x_var), self.loss(out, self.x_var*-1)])

        diff_squared = np.subtract(torch_to_np(out), self.x)**2
        sum = np.sum(diff_squared)
        mse = sum/(np.prod(diff_squared.shape))
        diff_squared = np.subtract(torch_to_np(out), self.x*-1)**2
        sum = np.sum(diff_squared)
        mse2 = sum/(np.prod(diff_squared.shape))

        total_loss_img = np.min([mse,mse2])

        # Print / Save Progress
        if(save_flag):
            self.print_progress(out_np, total_loss, A_type, mn_ratio)

        self.iter += 1

        return total_loss, total_loss_img

    def optimize_setup(self):
        # Enable GPU usage
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Misc. Params
        self.PLOT = False
        self.SAVE = True
        self.dim_div_by = 64

        # Optimization Params
        self.OPT_OVER = 'net'
        self.OPTIMIZER = 'adam'
        self.NET_TYPE = 'skip_depth6'  # one of skip_depth4|skip_depth2|UNET|ResNet

        self.LR = 0.01  # learning rate
        self.param_noise = False  # add noise to net params during optimization
        self.show_every = 100
        self.figsize = 5
        self.reg_noise_std = 0.03  # add noise to net input during optimization

        # Make copies
        self.net_input_saved = self.net_input.detach().clone()
        self.noise = self.net_input.detach().clone()

    def get_torch_vars(self, A_type):
        measurement_var = np_to_torch(self.measurement, False).type(self.dtype)
        x_var = np_to_torch(self.x, False).type(self.dtype)

        if (A_type == 'gaussian'):
            A_var = np_to_torch(self.A, False).type(self.dtype)
            real_A_var = None; img_A_var = None

        elif (A_type == 'c_gaussian' or A_type == 'DFT'  or A_type == '2d_DFT' or A_type == 'oversampled_2d_DFT'):
            real_A_var = np_to_torch(np.real(self.A), False).type(self.dtype)
            img_A_var = np_to_torch(np.imag(self.A), False).type(self.dtype)
            A_var = None

        else:
            raise NotImplementedError

        return measurement_var, A_var, x_var, real_A_var, img_A_var

    def print_progress(self, image, total_loss, A_type, mn_ratio):

        print('Iteration %05d    Loss %f' % (self.iter, total_loss.item()), '\r', end='')
        name = "reconst_iter: " + str(self.iter) + ".png"
        img_directory = self.get_full_directory(A_type)
        full_directory = img_directory + "mn_ratio="+str(mn_ratio)+"/"
        if not os.path.exists(full_directory):
            os.makedirs(full_directory)

        if self.PLOT and self.iter % self.show_every == 0:
            plot_image_grid([np.clip(image, 0, 1)], factor=self.figsize,
                            nrow=1, save=True, directory=full_directory, filename=name)

        if self.SAVE and self.iter % self.show_every == 0:
            plt.figure(figsize=(50, 50))
            plt.imshow(image[0], cmap='gray', interpolation='lanczos')
            plt.savefig(full_directory + name)
            plt.close('all')

    def plot_loss_curves(self, losses, img_losses, A_type, mn_ratio): #TODO: not sure if it's plotting MSE or just E
        if self.SAVE:
            img_directory = self.get_full_directory(A_type)
            full_directory = img_directory + "mn_ratio=" + str(mn_ratio) + "/"
            if not os.path.exists(full_directory):
                os.makedirs(full_directory)

            xs = np.arange(len(losses))+1
            plt.figure()
            plt.plot(xs, losses)
            plt.title('Measurement Loss vs. Time'); plt.xlabel('Iteration'); plt.ylabel('MSE (per pixel)')
            plt.savefig(full_directory + "Losses")

            plt.figure()
            plt.plot(xs, img_losses)
            plt.title('Image Loss vs. Time'); plt.xlabel('Iteration'); plt.ylabel('MSE (per pixel)')
            plt.savefig(full_directory + "Img_losses")
            plt.close('all')

    def get_full_directory(self, A_type):
        full_directory = self.plot_dir + str(A_type) + '/' + self.image_name + '/'

        if not os.path.exists(full_directory):
            os.makedirs(full_directory)

        return full_directory


class MNIST_DIP_net(DIP_net):

    def __init__(self, image, in_shape, img_name):
        super(MNIST_DIP_net, self).__init__(image, in_shape, img_name)

    def build_network(self):

        net = DCGAN_MNIST(nz=self.in_shape, nc=1, ngf=128)
        net = net.type(self.dtype)

        net_input = Variable(torch.zeros(1 * self.in_shape).type(self.dtype).view(1, self.in_shape, 1, 1))
        net_input.data.normal_().type(self.dtype)

        return net_input, net

class DCGAN_MNIST(nn.Module):
    def __init__(self, nz, ngf=128, output_size=28, nc=1, num_measurements=10):
        super(DCGAN_MNIST, self).__init__()
        self.nc = nc
        self.output_size = output_size

        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 3, 1, 1, bias=False)

    def forward(self, x):
        input_size = x.size()
        x = F.upsample(F.relu(self.bn1(self.conv1(x))), scale_factor=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.upsample(F.relu(self.bn3(self.conv3(x))), scale_factor=2)
        x = F.upsample(F.relu(self.bn4(self.conv4(x))), scale_factor=2)
        x = F.tanh(self.conv5(x, output_size=(-1, self.nc, self.output_size, self.output_size)))

        return x


#------------------------------------------------------------------------------
# Under Development
#-----------------------------

