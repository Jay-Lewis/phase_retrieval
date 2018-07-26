import numpy as np
import torch
import matplotlib as plt
import torch.optim
from src.models.skip import skip
from src.utils.inpainting_utils import *
from src.utils.utils import *


class DIP_net(object):

    def __init__(self, image, in_shape):
        self.dtype = torch.cuda.FloatTensor
        self.in_shape = in_shape
        self.x = image
        self.channels, self.n1, self.n2 = image.shape
        self.net_input, self.net = self.build_network()
        self.loss = self.define_loss()
        self.iter = 0
        self.plot_dir = "../plots/DIP/"

    def build_network(self):
        INPUT = 'meshgrid'
        input_depth = 2  # depth of input noise
        output_depth = self.channels
        pad = 'reflection'  # 'zero'

        num1 = 5  # 5
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

    def reconstruct(self, A_type, reconstruct_mode, mn_ratio, num_iter):
        self.iter = 0
        self.A = self.get_A(mn_ratio, A_type)
        self.measurement = self.get_measurement(A_type, reconstruct_mode)
        self.measurement_var, self.A_var, self.x_var, self.real_A_var, self.img_A_var = self.get_torch_vars(A_type)
        self.optimize_setup()
        p = get_params(self.OPT_OVER, self.net, self.net_input)
        optimize_function_params = (A_type, reconstruct_mode)
        losses, img_losses = optimize(self.OPTIMIZER, p, self.optimize_function,
                                      self.LR, num_iter, *optimize_function_params)

        return losses, img_losses

    def get_A(self, mn_ratio, A_type):
        m = int(self.n1 * mn_ratio)

        if A_type == "gaussian":
            mu = 0.0
            sigma = 1.0/m
            A = mu + sigma * np.random.randn(m, self.channels, self.n1)

        elif A_type == "DFT":
            if(self.channels == 1):
                A = np.fft.fft(np.eye(self.n1))  # might need to divide by sqrt(n) *******
                A = A.reshape(self.n1, 1, self.n1)

            else:
                raise ValueError('not yet implemented for images with more than one color channel')

        elif A_type == "c_gaussian":
            mu = 0.0
            sigma = 1.0 / float(2 * m) # the variance of each element isn't equal to the normal gaussian case
            A_real = mu + sigma * np.random.randn(m, self.channels, self.n1)
            A_img = mu + sigma * np.random.randn(m, self.channels, self.n1)
            A = A_real + 1j * A_img

        else:
            raise NotImplementedError

        return A

    def get_measurement(self, A_type, reconstruct_mode):

        if(reconstruct_mode == 'linear'):
            if (A_type == 'gaussian'):
                y = np.tensordot(self.A, self.x, 2)

            elif (A_type == 'c_gaussian' or A_type == 'DFT'):
                raise NotImplementedError

            else:
                raise NotImplementedError

        elif(reconstruct_mode == 'phase-retrieval'):
            y = np.tensordot(self.A, self.x, 2)
            y = np.abs(y)

        else:
            raise NotImplementedError

        return y

    def get_measurement_hat(self, out, A_type, reconstruct_mode):
        if (A_type == 'gaussian'):
            if (reconstruct_mode == 'linear'):
                measurement_hat = tensordot_pytorch(self.A_var, out, axes=2)

            elif (reconstruct_mode == 'phase-retrieval'):
                measurement_hat = tensordot_pytorch(self.A_var, out, axes=2)
                measurement_hat = measurement_hat.abs()
            else:
                raise NotImplementedError

        elif (A_type == 'c_gaussian' or A_type == 'DFT'):
            if (reconstruct_mode == 'linear'):
                raise NotImplementedError

            elif (reconstruct_mode == 'phase-retrieval'):
                measurement_hat = (tensordot_pytorch(self.real_A_var, out, axes=2) ** 2
                                   + tensordot_pytorch(self.img_A_var, out, axes=2) ** 2) ** (1 / 2)
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
        plt.savefig(self.plot_dir + "m sweep comparison.png")
        plt.show()

    def sweep_m_plot(self, mn_ratios, A_type, reconstruct_mode, num_iter, plot_flag=True, save_flag=True):
        final_losses = []
        for mn_ratio in mn_ratios:
            _, img_losses = self.reconstruct(A_type, reconstruct_mode, mn_ratio, num_iter)
            final_loss = img_losses[-1]
            final_losses.append(final_loss)

        xs = mn_ratios*self.n1
        plt.plot(xs.astype(int), final_losses, label=str(A_type))
        plt.xlabel('m (number of measurements)')
        plt.ylabel('mse (per pixel)')

        if(save_flag):  #NOT WORKING
            plt.title('m sweep ' + str(A_type))
            plt.savefig(self.plot_dir+"m_sweep"+str(A_type)+".png")

        if (plot_flag):
            plt.title('m sweep ' + str(A_type))
            plt.show()

        return

    def optimize_function(self, A_type, reconstruct_mode):

        # Add noise to network parameters / network input
        if self.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        self.net_input = self.net_input_saved
        if self.reg_noise_std > 0:
            self.net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)

        net_input_np = torch_to_np(self.net_input)

        # Output of network
        out = self.net(self.net_input)
        out_np = torch_to_np(out)

        # Calculate measurement estimate |A*net_out|
        new_shape = tuple(out.shape[1:])  # eliminate unnecessary dimension
        out = out.view(new_shape)  # must reshape before tensordot
        measurement_hat = self.get_measurement_hat(out, A_type, reconstruct_mode)

        # Define Loss (||A*net_out|-|A*image||l2)
        total_loss = self.loss(measurement_hat, self.measurement_var)
        total_loss.backward()
        total_loss_img = self.loss(out, self.x_var)

        # Print Progress
        print('Iteration %05d    Loss %f' % (self.iter, total_loss.item()), '\r', end='')
        name = "reconst_iter: " + str(self.iter) + ".png"
        if self.PLOT and self.iter % self.show_every == 0:
            plot_image_grid([np.clip(out_np, 0, 1)], factor=self.figsize,
                            nrow=1, save=True, directory=self.plot_dir, filename=name)
        if self.SAVE and self.iter % self.show_every == 0:
            plt.figure(figsize=(50, 50))
            plt.imshow(out_np[0], cmap='gray', interpolation='lanczos')
            plt.savefig(self.plot_dir + name)

        self.iter += 1

        return total_loss, total_loss_img

    def optimize_setup(self):
        # Enable GPU usage
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        # Misc. Params
        self.PLOT = True
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

        elif (A_type == 'c_gaussian' or A_type == 'DFT'):
            real_A_var = np_to_torch(np.real(self.A), False).type(self.dtype)
            img_A_var = np_to_torch(np.imag(self.A), False).type(self.dtype)
            A_var = None

        else:
            raise NotImplementedError

        return measurement_var, A_var, x_var, real_A_var, img_A_var


#------------------------------------------------------------------------------
# Under Development
#-----------------------------

# class MNIST_DIP_net(DIP_net):
#
#     def __init__(self):
#         super(DIP_net, self).__init__()
#
#     def build_network(self):