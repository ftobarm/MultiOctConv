"""
Function that sum frequency-wise two outputs of a M-OctConv Layer 
inputs:
    a:  Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
        If there is no channels in a level the tensor is replace by the None object
    b: Equivalent of a.
"""
def oct_sum(a,b):
    x_h = safe_sum(a[0],b[0])
    x_l = safe_sum(a[1],b[1])
    x_ll = safe_sum(a[2],b[2])
    return (x_h, x_l, x_ll)
"""
Function to sum a single frequency of the output of M-OctConv Layer. Manage the posible troubles with None values.
inputs:
    a: channels to sum. Could be a tenosr or a None object
    b: channels to sum. Could be a tenosr or a None object
"""
def safe_sum(a, b):
    if b is None:
        return a
    elif a is None:
        return b
    else:
        return a + b

"""
Function that print each frequency or they shape of the output of a M-OctConv Layer
inputs:
    x: Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
        If there is no channels in a level the tensor is replace by the None object
    end: Final char for the print
    shape: Boolean, if true print the shape of the features maps instead of the values
"""
def safe_print(x, end="\n", shape = False):
    s = ""
    for a in x:
        if a is None:
            s += "None | "
        elif shape:
            s += "{} | ".format(a.shape)
        else:
            s += "{} | ".format(a)
    print(s[:-2], end=end)

"""
Function applied the same function in each frequency of the output of a M-OctConv Layer
inputs:
    x:  A tensor or Tuple of size 3  of tensor wich represent high, medium and low frequency features maps of M-OctConv in that order.
        If there is no channels in a level the tensor is replace by the None object.
    func: function to be applied in each frequency
    func_args: arguments for the function
"""
def func_over_freq(x, func, func_args=None):
    if func is None:
        return x

    x_h, x_l, x_ll = x if type(x) is tuple else (x, None, None)
    if x_h  is not None:
        x_h = func(x_h) if func_args is None else func(x_h, **func_args)
    if x_l  is not None:
        x_l = func(x_l) if func_args is None else func(x_l, **func_args)
    if x_ll is not None:
        x_ll = func(x_ll) if func_args is None else func(x_ll, **func_args)
    return (x_h, x_l, x_ll)