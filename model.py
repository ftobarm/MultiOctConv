import torch
import MultiOctConv.utils as  octUtils

"""
Class that apply a Normalization layer in each frequency of a M-OctConv output
Inputs:
    norm_layer : Normalization layer to be applied in the channels
    norm_layer_args: Arguments of the normaization layer
    alpha channels: Number of channels in high frequency

"""
class MultiOctNorm(torch.nn.Module):
    def __init__(self, norm_layer, norm_layer_args, alpha_channels, beta_channels, gamma_channels):
        super(MultiOctNorm, self).__init__()
        if norm_layer is None:
            self.norm_h = None
            self.norm_l = None
            self.norm_ll = None
        else:
            self.norm_h = None if gamma_channels == 0 else norm_layer(gamma_channels, **norm_layer_args)
            self.norm_l = None if alpha_channels == 0 else norm_layer(alpha_channels, **norm_layer_args)
            self.norm_ll = None if beta_channels == 0 else norm_layer(beta_channels,  **norm_layer_args)

    #x :Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
    #    If there is no channels in a level the tensor is replace by the None object
    def forward(self, x):
        x_h, x_l, x_ll = x if type(x) is tuple else (x, None, None)
        if not self.norm_h  is None:
            x_h = None if x_h is None else self.norm_h(x_h)
        if not self.norm_l  is None:
            x_l = None if x_l is None else self.norm_l(x_l)
        if not self.norm_ll is None:
            x_ll = None if x_ll is None else self.norm_ll(x_ll)
        return (x_h, x_l, x_ll)
"""
Class that apply a Activation function  in each channels
Inputs:
    activation_function : Activation functin to be applied in the channels
    activation_function_args: Arguments of the Actuvation function
    alpha channels: Number of channels in high frequency

"""
class MultiOctActiv(torch.nn.Module):
    def set_activation_function(self, activation_function, activation_function_args):
        self.activation = None if activation_function is None else activation_function(**activation_function_args)

    def __init__(self, activation_function, activation_function_args):
        super(MultiOctActiv, self).__init__()
        self.activation = None if activation_function is None else activation_function(**activation_function_args)

    #x :Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
    #    If there is no channels in a level the tensor is replace by the None object
    def forward(self, x):
        return octUtils.func_over_freq(x, self.activation)


"""
Class that implemente the MultiOctaveConvolution that replace a traditional convolution

Inputs:
    in_channels :   Number of input channels for the convolution
    out_channels:   Number of output channels for the convolution
    kernel_size :   Kernel size for the convolution
    alpha_in :      Portion of channels that the input has in mid frequency
    alpha_out :     Portion of channels that the output should have in mid frequency
    beta_in :       Portion of channels that the input has in low frequency
    beta_out :      Portion of channels that the output should have in low frequency
    norm_layer :    Normalization layer to be applied AFTER the convolution
    norm_layer_args : Arguments for the normalization layer
    activation_function : Activation function to be applied AFTER the convolution
    activation_function_args : Arguments for the activation function
    conv :          Type of convolution to use (Conv1D,2D, deconv1d, etc.)
    conv_args: Extra arguments for the convolution (stride, bias, padding, etc)
    downsample: Function to be use to downsample the feature maps (MaxPool, AvgPool, etc.)
    upsample: Function to be use to upsample the feature maps (nearest, linear, ect.)
    print_x: Boolean to print x values in forward pass for debug

"""
class MultiOctaveConv(torch.nn.Module):
    
        
    def __init__(self, in_channels, out_channels, kernel_size, 
                alpha_in=0.25, alpha_out=0.25, beta_in=0.25,beta_out=0.25,
                norm_layer = None, norm_layer_args = {}, 
                activation_function = None, activation_function_args = {},
                conv= torch.nn.Conv2d, conv_args = {},
                downsample = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2),
                upsample = torch.nn.Upsample(scale_factor=2, mode='nearest'), print_x=False
                ):
        
        super(MultiOctaveConv, self).__init__()
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1 and 0 <= beta_in <= 1 and 0 <= beta_out <= 1, "Alphas should be in the interval from 0 to 1."
        assert 0 <= alpha_in + beta_in <= 1 and 0 <= alpha_out + beta_out <= 1, "Alphas_x + Beta_x should be in the interval from 0 to 1."
        self.print_x = print_x
        if type(kernel_size) is tuple:
            kernel_size_l = kernel_size[1]
            kernel_size = kernel_size[0]
        else:
            kernel_size_l = kernel_size

        ###to add bias to only one convolution level
        conv_args_bias = dict(conv_args)
        if "bias" not in conv_args:
            conv_args["bias"] = False 
            conv_args_bias["bias"] = True
        elif conv_args["bias"]:
            conv_args["bias"] = False 

        self.downsample = downsample
        self.upsample = upsample
        self.activation = MultiOctActiv(activation_function, activation_function_args)    
        alpha_in_channels = int(alpha_in * in_channels)
        beta_in_channels = int(beta_in * in_channels)
        gamma_in_channels = in_channels - alpha_in_channels -beta_in_channels
        
        self.alpha_out_channels = int(alpha_out * out_channels)
        self.beta_out_channels = int(beta_out * out_channels)
        self.gamma_out_channels = out_channels - self.alpha_out_channels -self.beta_out_channels

        self.norm_layer = MultiOctNorm(norm_layer, norm_layer_args, self.alpha_out_channels, self.beta_out_channels, self.gamma_out_channels)

        self.conv_h2h = None if gamma_in_channels == 0 or self.gamma_out_channels == 0 else \
                        conv(gamma_in_channels, self.gamma_out_channels, kernel_size, **conv_args_bias)
        self.conv_h2l = None if gamma_in_channels == 0 or self.alpha_out_channels == 0 else \
                        conv(gamma_in_channels, self.alpha_out_channels, kernel_size, **conv_args_bias)

        self.conv_l2h = None if alpha_in_channels == 0 or self.gamma_out_channels == 0 else \
                        conv(alpha_in_channels, self.gamma_out_channels, kernel_size_l, **conv_args)
        self.conv_l2l = None if alpha_in_channels == 0 or self.alpha_out_channels == 0 else \
                        conv(alpha_in_channels, self.alpha_out_channels, kernel_size_l, **conv_args)
        self.conv_l2ll = None if alpha_in_channels == 0 or self.beta_out_channels == 0 else \
                        conv(alpha_in_channels, self.beta_out_channels, kernel_size_l, **conv_args_bias)

        self.conv_ll2l = None if beta_in_channels == 0 or self.alpha_out_channels == 0 else \
                        conv(beta_in_channels, self.alpha_out_channels, kernel_size, **conv_args)
        self.conv_ll2ll = None if beta_in_channels == 0 or self.beta_out_channels == 0 else \
                        conv(beta_in_channels, self.beta_out_channels, kernel_size, **conv_args)

    #x:  A tensor or Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
    #    If there is no channels in a level the tensor is replace by the None object.
    def forward(self, x):
        if self.print_x:
            octUtils.safe_print(x, shape=True)
            print("*****")

        x_h, x_l, x_ll = x if type(x) is tuple else (x, None, None)
        
        out_x_h = None
        out_x_l = None
        out_x_ll = None

        if x_h is not None:
            x_h2h = self.conv_h2h(x_h)  if self.gamma_out_channels > 0 else None
            x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out_channels >0 else None
            out_x_h = x_h2h
            out_x_l = x_h2l

        if x_l is not None:
            x_l2h = self.upsample(self.conv_l2h(x_l)) if self.gamma_out_channels >0 else None
            x_l2l = self.conv_l2l(x_l) if self.alpha_out_channels > 0 else None     
            x_l2ll = self.downsample(self.conv_l2ll(x_l)) if self.beta_out_channels >0 else None
            out_x_h = octUtils.safe_sum(out_x_h,x_l2h)
            out_x_l = octUtils.safe_sum(out_x_l,x_l2l)
            out_x_ll = octUtils.safe_sum(out_x_ll,x_l2ll)

        if x_ll is not None:            
            x_ll2l = self.upsample(self.conv_ll2l(x_ll)) if self.alpha_out_channels >0 else None
            x_ll2ll = self.conv_ll2ll(x_ll) if self.beta_out_channels > 0 else None 
            out_x_l = octUtils.safe_sum(out_x_l,x_ll2l)
            out_x_ll = octUtils.safe_sum(out_x_ll,x_ll2ll)


        x =self.activation(self.norm_layer((out_x_h, out_x_l, out_x_ll)))
        if self.print_x:
            octUtils.safe_print(x, shape=True)
            print("----------------------------------------------------------------------")
        return self.activation(self.norm_layer((out_x_h, out_x_l, out_x_ll)))
