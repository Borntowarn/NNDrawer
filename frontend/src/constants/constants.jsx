const nodes = {
    "Functions": {
        "_sym_int": {
          "Doc": " SymInt-aware utility for int casting.\n\n    Args:\n        a (SymInt, SymFloat, or object): Object to cast\n    ",
          "Args": {
            "a": {
              "Type": null,
              "Default": null
            }
          }
        },
        "adaptive_avg_pool1d": {
          "Doc": "\nadaptive_avg_pool1d(input, output_size) -> Tensor\n\nApplies a 1D adaptive average pooling over an input signal composed of\nseveral input planes.\n\nSee :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.\n\nArgs:\n    output_size: the target output size (single integer)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "output_size": {
              "Type": null,
              "Default": null
            }
          }
        },
        "adaptive_avg_pool2d": {
          "Doc": "\n    Applies a 2D adaptive average pooling over an input signal composed of\n    several input planes.\n\n    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.\n\n    Args:\n        output_size: the target output size (single integer or\n            double-integer tuple)\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            }
          }
        },
        "adaptive_avg_pool3d": {
          "Doc": "\n    Applies a 3D adaptive average pooling over an input signal composed of\n    several input planes.\n\n    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.\n\n    Args:\n        output_size: the target output size (single integer or\n            triple-integer tuple)\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            }
          }
        },
        "adaptive_max_pool1d_with_indices": {
          "Doc": "Applies a 1D adaptive max pooling over an input signal composed of\n    several input planes.\n\n    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.\n\n    Args:\n        output_size: the target output size (single integer)\n        return_indices: whether to return pooling indices. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "adaptive_max_pool2d_with_indices": {
          "Doc": "Applies a 2D adaptive max pooling over an input signal composed of\n    several input planes.\n\n    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.\n\n    Args:\n        output_size: the target output size (single integer or\n            double-integer tuple)\n        return_indices: whether to return pooling indices. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "adaptive_max_pool3d_with_indices": {
          "Doc": "Applies a 3D adaptive max pooling over an input signal composed of\n    several input planes.\n\n    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.\n\n    Args:\n        output_size: the target output size (single integer or\n            triple-integer tuple)\n        return_indices: whether to return pooling indices. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "affine_grid": {
          "Doc": "Generates a 2D or 3D flow field (sampling grid), given a batch of\n    affine matrices :attr:`theta`.\n\n    .. note::\n        This function is often used in conjunction with :func:`grid_sample`\n        to build `Spatial Transformer Networks`_ .\n\n    Args:\n        theta (Tensor): input batch of affine matrices with shape\n            (:math:`N \\times 2 \\times 3`) for 2D or\n            (:math:`N \\times 3 \\times 4`) for 3D\n        size (torch.Size): the target output image size.\n            (:math:`N \\times C \\times H \\times W` for 2D or\n            :math:`N \\times C \\times D \\times H \\times W` for 3D)\n            Example: torch.Size((32, 3, 24, 24))\n        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``\n            to refer to the centers of the corner pixels rather than the image corners.\n            Refer to :func:`grid_sample` for a more complete description.\n            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`\n            with the same setting for this option.\n            Default: ``False``\n\n    Returns:\n        output (Tensor): output Tensor of size (:math:`N \\times H \\times W \\times 2`)\n\n    .. _`Spatial Transformer Networks`:\n        https://arxiv.org/abs/1506.02025\n\n    .. warning::\n        When ``align_corners = True``, the grid positions depend on the pixel\n        size relative to the input image size, and so the locations sampled by\n        :func:`grid_sample` will differ for the same input given at different\n        resolutions (that is, after being upsampled or downsampled).\n        The default behavior up to version 1.2.0 was ``align_corners = True``.\n        Since then, the default behavior has been changed to ``align_corners = False``,\n        in order to bring it in line with the default for :func:`interpolate`.\n    .. warning::\n        When ``align_corners = True``, 2D affine transforms on 1D data and\n        3D affine transforms on 2D data (that is, when one of the spatial\n        dimensions has unit size) are ill-defined, and not an intended use case.\n        This is not a problem when ``align_corners = False``.\n        Up to version 1.2.0, all grid points along a unit dimension were\n        considered arbitrarily to be at ``-1``.\n        From version 1.3.0, under ``align_corners = True`` all grid points\n        along a unit dimension are considered to be at ``0``\n        (the center of the input image).\n    ",
          "Args": {
            "theta": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size": {
              "Type": "typing.List[int]",
              "Default": null
            },
            "align_corners": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            }
          }
        },
        "alpha_dropout": {
          "Doc": "Applies alpha dropout to the input.\n\n    See :class:`~torch.nn.AlphaDropout` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "assert_int_or_pair": {
          "Doc": null,
          "Args": {
            "arg": {
              "Type": "typing.List[int]",
              "Default": null
            },
            "arg_name": {
              "Type": "<class 'str'>",
              "Default": null
            },
            "message": {
              "Type": "<class 'str'>",
              "Default": null
            }
          }
        },
        "avg_pool1d": {
          "Doc": "\navg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor\n\nApplies a 1D average pooling over an input signal composed of several\ninput planes.\n\nSee :class:`~torch.nn.AvgPool1d` for details and output shape.\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    kernel_size: the size of the window. Can be a single number or a\n      tuple `(kW,)`\n    stride: the stride of the window. Can be a single number or a tuple\n      `(sW,)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padW,)`. Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` to compute the\n        output shape. Default: ``False``\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation. Default: ``True``\n\nExamples::\n\n    >>> # pool of square window of size=3, stride=2\n    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)\n    >>> F.avg_pool1d(input, kernel_size=3, stride=2)\n    tensor([[[ 2.,  4.,  6.]]])\n\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "kernel_size": {
              "Type": null,
              "Default": null
            },
            "stride": {
              "Type": null,
              "Default": "None"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "ceil_mode": {
              "Type": null,
              "Default": "False"
            },
            "count_include_pad": {
              "Type": null,
              "Default": "True"
            }
          }
        },
        "avg_pool2d": {
          "Doc": "\navg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor\n\nApplies 2D average-pooling operation in :math:`kH \\times kW` regions by step size\n:math:`sH \\times sW` steps. The number of output features is equal to the number of\ninput planes.\n\nSee :class:`~torch.nn.AvgPool2d` for details and output shape.\n\nArgs:\n    input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    kernel_size: size of the pooling region. Can be a single number or a\n      tuple `(kH, kW)`\n    stride: stride of the pooling operation. Can be a single number or a\n      tuple `(sH, sW)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padH, padW)`. Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` in the formula\n        to compute the output shape. Default: ``False``\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation. Default: ``True``\n    divisor_override: if specified, it will be used as divisor, otherwise\n         size of the pooling region will be used. Default: None\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "kernel_size": {
              "Type": null,
              "Default": null
            },
            "stride": {
              "Type": null,
              "Default": "None"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "ceil_mode": {
              "Type": null,
              "Default": "False"
            },
            "count_include_pad": {
              "Type": null,
              "Default": "True"
            },
            "divisor_override": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "avg_pool3d": {
          "Doc": "\navg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor\n\nApplies 3D average-pooling operation in :math:`kT \\times kH \\times kW` regions by step\nsize :math:`sT \\times sH \\times sW` steps. The number of output features is equal to\n:math:`\\lfloor\\frac{\\text{input planes}}{sT}\\rfloor`.\n\nSee :class:`~torch.nn.AvgPool3d` for details and output shape.\n\nArgs:\n    input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iT \\times iH , iW)`\n    kernel_size: size of the pooling region. Can be a single number or a\n      tuple `(kT, kH, kW)`\n    stride: stride of the pooling operation. Can be a single number or a\n      tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padT, padH, padW)`, Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` in the formula\n        to compute the output shape\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation\n    divisor_override: if specified, it will be used as divisor, otherwise\n        size of the pooling region will be used. Default: None\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "kernel_size": {
              "Type": null,
              "Default": null
            },
            "stride": {
              "Type": null,
              "Default": "None"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "ceil_mode": {
              "Type": null,
              "Default": "False"
            },
            "count_include_pad": {
              "Type": null,
              "Default": "True"
            },
            "divisor_override": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "batch_norm": {
          "Doc": "Applies Batch Normalization for each channel across a batch of data.\n\n    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,\n    :class:`~torch.nn.BatchNorm3d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "running_mean": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "running_var": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "momentum": {
              "Type": "<class 'float'>",
              "Default": "0.1"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-05"
            }
          }
        },
        "bilinear": {
          "Doc": "\nbilinear(input1, input2, weight, bias=None) -> Tensor\n\nApplies a bilinear transformation to the incoming data:\n:math:`y = x_1^T A x_2 + b`\n\nShape:\n\n    - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\\text{in1\\_features}`\n      and :math:`*` means any number of additional dimensions.\n      All but the last dimension of the inputs should be the same.\n    - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\\text{in2\\_features}`\n    - weight: :math:`(\\text{out\\_features}, \\text{in1\\_features},\n      \\text{in2\\_features})`\n    - bias: :math:`(\\text{out\\_features})`\n    - output: :math:`(N, *, H_{out})` where :math:`H_{out}=\\text{out\\_features}`\n      and all but the last dimension are the same shape as the input.\n",
          "Args": {
            "input1": {
              "Type": null,
              "Default": null
            },
            "input2": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "binary_cross_entropy": {
          "Doc": "Function that measures the Binary Cross Entropy between the target and input\n    probabilities.\n\n    See :class:`~torch.nn.BCELoss` for details.\n\n    Args:\n        input: Tensor of arbitrary shape as probabilities.\n        target: Tensor of the same shape as input with values between 0 and 1.\n        weight (Tensor, optional): a manual rescaling weight\n                if provided it's repeated to match input tensor shape\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n    Examples::\n\n        >>> input = torch.randn(3, 2, requires_grad=True)\n        >>> target = torch.rand(3, 2, requires_grad=False)\n        >>> loss = F.binary_cross_entropy(torch.sigmoid(input), target)\n        >>> loss.backward()\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "binary_cross_entropy_with_logits": {
          "Doc": "Function that measures Binary Cross Entropy between target and input\n    logits.\n\n    See :class:`~torch.nn.BCEWithLogitsLoss` for details.\n\n    Args:\n        input: Tensor of arbitrary shape as unnormalized scores (often referred to as logits).\n        target: Tensor of the same shape as input with values between 0 and 1\n        weight (Tensor, optional): a manual rescaling weight\n            if provided it's repeated to match input tensor shape\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n        pos_weight (Tensor, optional): a weight of positive examples.\n                Must be a vector with length equal to the number of classes.\n\n    Examples::\n\n         >>> input = torch.randn(3, requires_grad=True)\n         >>> target = torch.empty(3).random_(2)\n         >>> loss = F.binary_cross_entropy_with_logits(input, target)\n         >>> loss.backward()\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "pos_weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            }
          }
        },
        "boolean_dispatch": {
          "Doc": "\n    Dispatches to either of 2 script functions based on a boolean argument.\n    In TorchScript, the boolean argument must be constant so that the correct\n    function to use can be determined at compile time.\n    ",
          "Args": {
            "arg_name": {
              "Type": null,
              "Default": null
            },
            "arg_index": {
              "Type": null,
              "Default": null
            },
            "default": {
              "Type": null,
              "Default": null
            },
            "if_true": {
              "Type": null,
              "Default": null
            },
            "if_false": {
              "Type": null,
              "Default": null
            },
            "module_name": {
              "Type": null,
              "Default": null
            },
            "func_name": {
              "Type": null,
              "Default": null
            }
          }
        },
        "celu": {
          "Doc": "celu(input, alpha=1., inplace=False) -> Tensor\n\n    Applies element-wise,\n    :math:`\\text{CELU}(x) = \\max(0,x) + \\min(0, \\alpha * (\\exp(x/\\alpha) - 1))`.\n\n    See :class:`~torch.nn.CELU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "alpha": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "channel_shuffle": {
          "Doc": "\nchannel_shuffle(input, groups) -> Tensor\n\nDivide the channels in a tensor of shape :math:`(*, C , H, W)`\ninto g groups and rearrange them as :math:`(*, C \\frac g, g, H, W)`,\nwhile keeping the original tensor shape.\n\nSee :class:`~torch.nn.ChannelShuffle` for details.\n\nArgs:\n    input (Tensor): the input tensor\n    groups (int): number of groups to divide channels in and rearrange.\n\nExamples::\n\n    >>> input = torch.randn(1, 4, 2, 2)\n    >>> print(input)\n    [[[[1, 2],\n       [3, 4]],\n      [[5, 6],\n       [7, 8]],\n      [[9, 10],\n       [11, 12]],\n      [[13, 14],\n       [15, 16]],\n     ]]\n    >>> output = torch.nn.functional.channel_shuffle(input, 2)\n    >>> print(output)\n    [[[[1, 2],\n       [3, 4]],\n      [[9, 10],\n       [11, 12]],\n      [[5, 6],\n       [7, 8]],\n      [[13, 14],\n       [15, 16]],\n     ]]\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "groups": {
              "Type": null,
              "Default": null
            }
          }
        },
        "conv1d": {
          "Doc": "\nconv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 1D convolution over an input signal composed of several input\nplanes.\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.Conv1d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\nNote:\n    This operator supports complex data types i.e. ``complex32, complex64, complex128``.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: ``None``\n    stride: the stride of the convolving kernel. Can be a single number or\n      a one-element tuple `(sW,)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},\n      single number or a one-element tuple `(padW,)`. Default: 0\n      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n      the input so the output has the same shape as the input. However, this mode\n      doesn't support any stride values other than 1.\n\n      .. warning::\n          For ``padding='same'``, if the ``weight`` is even-length and\n          ``dilation`` is odd in any dimension, a full :func:`pad` operation\n          may be needed internally. Lowering performance.\n    dilation: the spacing between kernel elements. Can be a single number or\n      a one-element tuple `(dW,)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by\n      the number of groups. Default: 1\n\nExamples::\n\n    >>> inputs = torch.randn(33, 16, 30)\n    >>> filters = torch.randn(20, 16, 5)\n    >>> F.conv1d(inputs, filters)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "conv2d": {
          "Doc": "\nconv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 2D convolution over an input image composed of several input\nplanes.\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.Conv2d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\nNote:\n    This operator supports complex data types i.e. ``complex32, complex64, complex128``.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kH , kW)`\n    bias: optional bias tensor of shape :math:`(\\text{out\\_channels})`. Default: ``None``\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple `(sH, sW)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},\n      single number or a tuple `(padH, padW)`. Default: 0\n      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n      the input so the output has the same shape as the input. However, this mode\n      doesn't support any stride values other than 1.\n\n      .. warning::\n          For ``padding='same'``, if the ``weight`` is even-length and\n          ``dilation`` is odd in any dimension, a full :func:`pad` operation\n          may be needed internally. Lowering performance.\n\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dH, dW)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n\nExamples::\n\n    >>> # With square kernels and equal stride\n    >>> filters = torch.randn(8, 4, 3, 3)\n    >>> inputs = torch.randn(1, 4, 5, 5)\n    >>> F.conv2d(inputs, filters, padding=1)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "conv3d": {
          "Doc": "\nconv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 3D convolution over an input image composed of several input\nplanes.\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.Conv3d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\nNote:\n    This operator supports complex data types i.e. ``complex32, complex64, complex128``.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iT , iH , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kT , kH , kW)`\n    bias: optional bias tensor of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple `(sT, sH, sW)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a string {'valid', 'same'},\n      single number or a tuple `(padT, padH, padW)`. Default: 0\n      ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n      the input so the output has the same shape as the input. However, this mode\n      doesn't support any stride values other than 1.\n\n      .. warning::\n          For ``padding='same'``, if the ``weight`` is even-length and\n          ``dilation`` is odd in any dimension, a full :func:`pad` operation\n          may be needed internally. Lowering performance.\n\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dT, dH, dW)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by\n      the number of groups. Default: 1\n\nExamples::\n\n    >>> filters = torch.randn(33, 16, 3, 3, 3)\n    >>> inputs = torch.randn(20, 16, 50, 10, 20)\n    >>> F.conv3d(inputs, filters)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "conv_tbc": {
          "Doc": "\nApplies a 1-dimensional sequence convolution over an input sequence.\nInput and output dimensions are (Time, Batch, Channels) - hence TBC.\n\nArgs:\n    input: input tensor of shape :math:`(\\text{sequence length} \\times batch \\times \\text{in\\_channels})`\n    weight: filter of shape (:math:`\\text{kernel width} \\times \\text{in\\_channels} \\times \\text{out\\_channels}`)\n    bias: bias of shape (:math:`\\text{out\\_channels}`)\n    pad: number of timesteps to pad. Default: 0\n",
          "Args": {
            "Time": {
              "Type": null,
              "Default": null
            },
            "Batch": {
              "Type": null,
              "Default": null
            },
            "Channels": {
              "Type": null,
              "Default": null
            }
          }
        },
        "conv_transpose1d": {
          "Doc": "\nconv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 1D transposed convolution operator over an input signal\ncomposed of several input planes, sometimes also called \"deconvolution\".\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.ConvTranspose1d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sW,)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padW,)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple ``(dW,)``. Default: 1\n\nExamples::\n\n    >>> inputs = torch.randn(20, 16, 50)\n    >>> weights = torch.randn(16, 33, 5)\n    >>> F.conv_transpose1d(inputs, weights)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "output_padding": {
              "Type": null,
              "Default": "0"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "conv_transpose2d": {
          "Doc": "\nconv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 2D transposed convolution operator over an input image\ncomposed of several input planes, sometimes also called \"deconvolution\".\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.ConvTranspose2d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kH , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sH, sW)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padH, padW)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.\n      Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple ``(dH, dW)``. Default: 1\n\nExamples::\n\n    >>> # With square kernels and equal stride\n    >>> inputs = torch.randn(1, 4, 5, 5)\n    >>> weights = torch.randn(4, 8, 3, 3)\n    >>> F.conv_transpose2d(inputs, weights, padding=1)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "output_padding": {
              "Type": null,
              "Default": "0"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "conv_transpose3d": {
          "Doc": "\nconv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 3D transposed convolution operator over an input image\ncomposed of several input planes, sometimes also called \"deconvolution\"\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nSee :class:`~torch.nn.ConvTranspose3d` for details and output shape.\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iT , iH , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kT , kH , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sT, sH, sW)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padT, padH, padW)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple\n      ``(out_padT, out_padH, out_padW)``. Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dT, dH, dW)`. Default: 1\n\nExamples::\n\n    >>> inputs = torch.randn(20, 16, 50, 10, 20)\n    >>> weights = torch.randn(16, 33, 3, 3, 3)\n    >>> F.conv_transpose3d(inputs, weights)\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            },
            "stride": {
              "Type": null,
              "Default": "1"
            },
            "padding": {
              "Type": null,
              "Default": "0"
            },
            "output_padding": {
              "Type": null,
              "Default": "0"
            },
            "groups": {
              "Type": null,
              "Default": "1"
            },
            "dilation": {
              "Type": null,
              "Default": "1"
            }
          }
        },
        "cosine_embedding_loss": {
          "Doc": "cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.CosineEmbeddingLoss` for details.\n    ",
          "Args": {
            "input1": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "input2": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "0"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "cosine_similarity": {
          "Doc": "\ncosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor\n\nReturns cosine similarity between ``x1`` and ``x2``, computed along dim. ``x1`` and ``x2`` must be broadcastable\nto a common shape. ``dim`` refers to the dimension in this common shape. Dimension ``dim`` of the output is\nsqueezed (see :func:`torch.squeeze`), resulting in the\noutput tensor having 1 fewer dimension.\n\n.. math ::\n    \\text{similarity} = \\dfrac{x_1 \\cdot x_2}{\\max(\\Vert x_1 \\Vert _2 \\cdot \\Vert x_2 \\Vert _2, \\epsilon)}\n\nSupports :ref:`type promotion <type-promotion-doc>`.\n\nArgs:\n    x1 (Tensor): First input.\n    x2 (Tensor): Second input.\n    dim (int, optional): Dimension along which cosine similarity is computed. Default: 1\n    eps (float, optional): Small value to avoid division by zero.\n        Default: 1e-8\n\nExample::\n\n    >>> input1 = torch.randn(100, 128)\n    >>> input2 = torch.randn(100, 128)\n    >>> output = F.cosine_similarity(input1, input2)\n    >>> print(output)\n",
          "Args": {
            "x1": {
              "Type": null,
              "Default": null
            },
            "x2": {
              "Type": null,
              "Default": null
            },
            "dim": {
              "Type": null,
              "Default": "1"
            },
            "eps": {
              "Type": null,
              "Default": "1e-8"
            }
          }
        },
        "cross_entropy": {
          "Doc": "This criterion computes the cross entropy loss between input logits and target.\n\n    See :class:`~torch.nn.CrossEntropyLoss` for details.\n\n    Args:\n        input (Tensor) : Predicted unnormalized logits;\n            see Shape section below for supported shapes.\n        target (Tensor) : Ground truth class indices or class probabilities;\n            see Shape section below for supported shapes.\n        weight (Tensor, optional): a manual rescaling weight given to each\n            class. If given, has to be a Tensor of size `C`\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        ignore_index (int, optional): Specifies a target value that is ignored\n            and does not contribute to the input gradient. When :attr:`size_average` is\n            ``True``, the loss is averaged over non-ignored targets. Note that\n            :attr:`ignore_index` is only applicable when the target contains class indices.\n            Default: -100\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n        label_smoothing (float, optional): A float in [0.0, 1.0]. Specifies the amount\n            of smoothing when computing the loss, where 0.0 means no smoothing. The targets\n            become a mixture of the original ground truth and a uniform distribution as described in\n            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.\n\n    Shape:\n        - Input: Shape :math:`(C)`, :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \\geq 1`\n          in the case of `K`-dimensional loss.\n        - Target: If containing class indices, shape :math:`()`, :math:`(N)` or :math:`(N, d_1, d_2, ..., d_K)` with\n          :math:`K \\geq 1` in the case of K-dimensional loss where each value should be between :math:`[0, C)`.\n          If containing class probabilities, same shape as the input and each value should be between :math:`[0, 1]`.\n\n        where:\n\n        .. math::\n            \\begin{aligned}\n                C ={} & \\text{number of classes} \\\\\n                N ={} & \\text{batch size} \\\\\n            \\end{aligned}\n\n    Examples::\n\n        >>> # Example of target with class indices\n        >>> input = torch.randn(3, 5, requires_grad=True)\n        >>> target = torch.randint(5, (3,), dtype=torch.int64)\n        >>> loss = F.cross_entropy(input, target)\n        >>> loss.backward()\n        >>>\n        >>> # Example of target with class probabilities\n        >>> input = torch.randn(3, 5, requires_grad=True)\n        >>> target = torch.randn(3, 5).softmax(dim=1)\n        >>> loss = F.cross_entropy(input, target)\n        >>> loss.backward()\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "ignore_index": {
              "Type": "<class 'int'>",
              "Default": "-100"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "label_smoothing": {
              "Type": "<class 'float'>",
              "Default": "0.0"
            }
          }
        },
        "ctc_loss": {
          "Doc": "The Connectionist Temporal Classification loss.\n\n    See :class:`~torch.nn.CTCLoss` for details.\n\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n\n    Args:\n        log_probs: :math:`(T, N, C)` or :math:`(T, C)` where `C = number of characters in alphabet including blank`,\n            `T = input length`, and `N = batch size`.\n            The logarithmized probabilities of the outputs\n            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).\n        targets: :math:`(N, S)` or `(sum(target_lengths))`.\n            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.\n        input_lengths: :math:`(N)` or :math:`()`.\n            Lengths of the inputs (must each be :math:`\\leq T`)\n        target_lengths: :math:`(N)` or :math:`()`.\n            Lengths of the targets\n        blank (int, optional):\n            Blank label. Default :math:`0`.\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the output losses will be divided by the target lengths and\n            then the mean over the batch is taken, ``'sum'``: the output will be\n            summed. Default: ``'mean'``\n        zero_infinity (bool, optional):\n            Whether to zero infinite losses and the associated gradients.\n            Default: ``False``\n            Infinite losses mainly occur when the inputs are too short\n            to be aligned to the targets.\n\n    Example::\n\n        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()\n        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)\n        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)\n        >>> target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)\n        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)\n        >>> loss.backward()\n    ",
          "Args": {
            "log_probs": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "targets": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "input_lengths": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target_lengths": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "blank": {
              "Type": "<class 'int'>",
              "Default": "0"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "zero_infinity": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "dropout": {
          "Doc": "\n    During training, randomly zeroes some of the elements of the input\n    tensor with probability :attr:`p` using samples from a Bernoulli\n    distribution.\n\n    See :class:`~torch.nn.Dropout` for details.\n\n    Args:\n        p: probability of an element to be zeroed. Default: 0.5\n        training: apply dropout if is ``True``. Default: ``True``\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "dropout1d": {
          "Doc": "\n    Randomly zero out entire channels (a channel is a 1D feature map,\n    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the\n    batched input is a 1D tensor :math:`\\text{input}[i, j]`) of the input tensor).\n    Each channel will be zeroed out independently on every forward call with\n    probability :attr:`p` using samples from a Bernoulli distribution.\n\n    See :class:`~torch.nn.Dropout1d` for details.\n\n    Args:\n        p: probability of a channel to be zeroed. Default: 0.5\n        training: apply dropout if is ``True``. Default: ``True``\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "dropout2d": {
          "Doc": "\n    Randomly zero out entire channels (a channel is a 2D feature map,\n    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the\n    batched input is a 2D tensor :math:`\\text{input}[i, j]`) of the input tensor).\n    Each channel will be zeroed out independently on every forward call with\n    probability :attr:`p` using samples from a Bernoulli distribution.\n\n    See :class:`~torch.nn.Dropout2d` for details.\n\n    Args:\n        p: probability of a channel to be zeroed. Default: 0.5\n        training: apply dropout if is ``True``. Default: ``True``\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "dropout3d": {
          "Doc": "\n    Randomly zero out entire channels (a channel is a 3D feature map,\n    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the\n    batched input is a 3D tensor :math:`\\text{input}[i, j]`) of the input tensor).\n    Each channel will be zeroed out independently on every forward call with\n    probability :attr:`p` using samples from a Bernoulli distribution.\n\n    See :class:`~torch.nn.Dropout3d` for details.\n\n    Args:\n        p: probability of a channel to be zeroed. Default: 0.5\n        training: apply dropout if is ``True``. Default: ``True``\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "elu": {
          "Doc": "Applies the Exponential Linear Unit (ELU) function element-wise.\n\n    See :class:`~torch.nn.ELU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "alpha": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "embedding": {
          "Doc": "A simple lookup table that looks up embeddings in a fixed dictionary and size.\n\n    This module is often used to retrieve word embeddings using indices.\n    The input to the module is a list of indices, and the embedding matrix,\n    and the output is the corresponding word embeddings.\n\n    See :class:`torch.nn.Embedding` for more details.\n\n    Args:\n        input (LongTensor): Tensor containing indices into the embedding matrix\n        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,\n            and number of columns equal to the embedding size\n        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the gradient;\n                                     therefore, the embedding vector at :attr:`padding_idx` is not updated during training,\n                                     i.e. it remains as a fixed \"pad\".\n        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`\n                                    is renormalized to have norm :attr:`max_norm`.\n                                    Note: this will modify :attr:`weight` in-place.\n        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.\n        scale_grad_by_freq (bool, optional): If given, this will scale gradients by the inverse of frequency of\n                                                the words in the mini-batch. Default ``False``.\n        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under\n                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.\n\n    Shape:\n        - Input: LongTensor of arbitrary shape containing the indices to extract\n        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,\n          where V = maximum index + 1 and embedding_dim = the embedding size\n        - Output: `(*, embedding_dim)`, where `*` is the input shape\n\n    Examples::\n\n        >>> # a batch of 2 samples of 4 indices each\n        >>> input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])\n        >>> # an embedding matrix containing 10 tensors of size 3\n        >>> embedding_matrix = torch.rand(10, 3)\n        >>> # xdoctest: +IGNORE_WANT(\"non-deterministic\")\n        >>> F.embedding(input, embedding_matrix)\n        tensor([[[ 0.8490,  0.9625,  0.6753],\n                 [ 0.9666,  0.7761,  0.6108],\n                 [ 0.6246,  0.9751,  0.3618],\n                 [ 0.4161,  0.2419,  0.7383]],\n\n                [[ 0.6246,  0.9751,  0.3618],\n                 [ 0.0237,  0.7794,  0.0528],\n                 [ 0.9666,  0.7761,  0.6108],\n                 [ 0.3385,  0.8612,  0.1867]]])\n\n        >>> # example with padding_idx\n        >>> weights = torch.rand(10, 3)\n        >>> weights[0, :].zero_()\n        >>> embedding_matrix = weights\n        >>> input = torch.tensor([[0, 2, 0, 5]])\n        >>> F.embedding(input, embedding_matrix, padding_idx=0)\n        tensor([[[ 0.0000,  0.0000,  0.0000],\n                 [ 0.5609,  0.5384,  0.8720],\n                 [ 0.0000,  0.0000,  0.0000],\n                 [ 0.6262,  0.2438,  0.7471]]])\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "padding_idx": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            },
            "max_norm": {
              "Type": "typing.Optional[float]",
              "Default": "None"
            },
            "norm_type": {
              "Type": "<class 'float'>",
              "Default": "2.0"
            },
            "scale_grad_by_freq": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "sparse": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "embedding_bag": {
          "Doc": "Computes sums, means or maxes of `bags` of embeddings, without instantiating the\n    intermediate embeddings.\n\n    See :class:`torch.nn.EmbeddingBag` for more details.\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n\n    Args:\n        input (LongTensor): Tensor containing bags of indices into the embedding matrix\n        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,\n            and number of columns equal to the embedding size\n        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines\n                             the starting index position of each bag (sequence) in :attr:`input`.\n        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`\n                                    is renormalized to have norm :attr:`max_norm`.\n                                    Note: this will modify :attr:`weight` in-place.\n        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.\n                                     Default ``2``.\n        scale_grad_by_freq (bool, optional): if given, this will scale gradients by the inverse of frequency of\n                                                the words in the mini-batch. Default ``False``.\n                                                Note: this option is not supported when ``mode=\"max\"``.\n        mode (str, optional): ``\"sum\"``, ``\"mean\"`` or ``\"max\"``. Specifies the way to reduce the bag.\n                                 Default: ``\"mean\"``\n        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under\n                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.\n                                 Note: this option is not supported when ``mode=\"max\"``.\n        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None\n            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`\n            must have exactly the same shape as input and is treated as having the same\n            :attr:`offsets`, if those are not None.\n\n        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.\n            The last element is the size of the input, or the ending index position of the last bag (sequence).\n\n        padding_idx (int, optional): If specified, the entries at :attr:`padding_idx` do not contribute to the\n                                     gradient; therefore, the embedding vector at :attr:`padding_idx` is not updated\n                                     during training, i.e. it remains as a fixed \"pad\". Note that the embedding\n                                     vector at :attr:`padding_idx` is excluded from the reduction.\n\n    Shape:\n        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)\n\n          - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)\n            each of fixed length ``N``, and this will return ``B`` values aggregated in a way\n            depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.\n\n          - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of\n            multiple bags (sequences). :attr:`offsets` is required to be a 1D tensor containing\n            the starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets`\n            of shape `(B)`, :attr:`input` will be viewed as having ``B`` bags.\n            Empty bags (i.e., having 0-length) will have returned vectors filled by zeros.\n\n        - :attr:`weight` (Tensor): the learnable weights of the module of shape `(num_embeddings, embedding_dim)`\n\n        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as :attr:`input`.\n\n        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`\n\n    Examples::\n\n        >>> # an Embedding module containing 10 tensors of size 3\n        >>> embedding_matrix = torch.rand(10, 3)\n        >>> # a batch of 2 samples of 4 indices each\n        >>> input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])\n        >>> offsets = torch.tensor([0, 4])\n        >>> # xdoctest: +IGNORE_WANT(\"non-deterministic\")\n        >>> F.embedding_bag(input, embedding_matrix, offsets)\n        tensor([[ 0.3397,  0.3552,  0.5545],\n                [ 0.5893,  0.4386,  0.5882]])\n\n        >>> # example with padding_idx\n        >>> embedding_matrix = torch.rand(10, 3)\n        >>> input = torch.tensor([2, 2, 2, 2, 4, 3, 2, 9])\n        >>> offsets = torch.tensor([0, 4])\n        >>> F.embedding_bag(input, embedding_matrix, offsets, padding_idx=2, mode='sum')\n        tensor([[ 0.0000,  0.0000,  0.0000],\n                [-0.7082,  3.2145, -2.6251]])\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "offsets": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "max_norm": {
              "Type": "typing.Optional[float]",
              "Default": "None"
            },
            "norm_type": {
              "Type": "<class 'float'>",
              "Default": "2"
            },
            "scale_grad_by_freq": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "mode": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "sparse": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "per_sample_weights": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "include_last_offset": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "padding_idx": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            }
          }
        },
        "feature_alpha_dropout": {
          "Doc": "\n    Randomly masks out entire channels (a channel is a feature map,\n    e.g. the :math:`j`-th channel of the :math:`i`-th sample in the batch input\n    is a tensor :math:`\\text{input}[i, j]`) of the input tensor). Instead of\n    setting activations to zero, as in regular Dropout, the activations are set\n    to the negative saturation value of the SELU activation function.\n\n    Each element will be masked independently on every forward call with\n    probability :attr:`p` using samples from a Bernoulli distribution.\n    The elements to be masked are randomized on every forward call, and scaled\n    and shifted to maintain zero mean and unit variance.\n\n    See :class:`~torch.nn.FeatureAlphaDropout` for details.\n\n    Args:\n        p: dropout probability of a channel to be zeroed. Default: 0.5\n        training: apply dropout if is ``True``. Default: ``True``\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "0.5"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "fold": {
          "Doc": "Combines an array of sliding local blocks into a large containing\n    tensor.\n\n    .. warning::\n        Currently, only unbatched (3D) or batched (4D) image-like output tensors are supported.\n\n    See :class:`torch.nn.Fold` for details\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "output_size": {
              "Type": "None",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "dilation": {
              "Type": "None",
              "Default": "1"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "stride": {
              "Type": "None",
              "Default": "1"
            }
          }
        },
        "fractional_max_pool2d_with_indices": {
          "Doc": "Applies 2D fractional max pooling over an input signal composed of several input planes.\n\n    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham\n\n    The max-pooling operation is applied in :math:`kH \\times kW` regions by a stochastic\n    step size determined by the target output size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        kernel_size: the size of the window to take a max over.\n                     Can be a single number :math:`k` (for a square kernel of :math:`k \\times k`)\n                     or a tuple `(kH, kW)`\n        output_size: the target output size of the image of the form :math:`oH \\times oW`.\n                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH \\times oH`\n        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.\n                      This has to be a number or tuple in the range (0, 1)\n        return_indices: if ``True``, will return the indices along with the outputs.\n                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.\n\n    Examples::\n        >>> input = torch.randn(20, 16, 50, 32)\n        >>> # pool of square window of size=3, and target output size 13x12\n        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))\n        >>> # pool of square window and target output size being half of input image size\n        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))\n\n    .. _Fractional MaxPooling:\n        http://arxiv.org/abs/1412.6071\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "output_size": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "output_ratio": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "_random_samples": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            }
          }
        },
        "fractional_max_pool3d_with_indices": {
          "Doc": "Applies 3D fractional max pooling over an input signal composed of several input planes.\n\n    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham\n\n    The max-pooling operation is applied in :math:`kT \\times kH \\times kW` regions by a stochastic\n    step size determined by the target output size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        kernel_size: the size of the window to take a max over.\n                     Can be a single number :math:`k` (for a square kernel of :math:`k \\times k \\times k`)\n                     or a tuple `(kT, kH, kW)`\n        output_size: the target output size of the form :math:`oT \\times oH \\times oW`.\n                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output\n                     :math:`oH \\times oH \\times oH`\n        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.\n                      This has to be a number or tuple in the range (0, 1)\n        return_indices: if ``True``, will return the indices along with the outputs.\n                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.\n\n    Shape:\n        - Input: :math:`(N, C, T_{in}, H_{in}, W_{in})` or :math:`(C, T_{in}, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, T_{out}, H_{out}, W_{out})` or :math:`(C, T_{out}, H_{out}, W_{out})`, where\n          :math:`(T_{out}, H_{out}, W_{out})=\\text{output\\_size}` or\n          :math:`(T_{out}, H_{out}, W_{out})=\\text{output\\_ratio} \\times (T_{in}, H_{in}, W_{in})`\n\n    Examples::\n        >>> input = torch.randn(20, 16, 50, 32, 16)\n        >>> # pool of cubic window of size=3, and target output size 13x12x11\n        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))\n        >>> # pool of cubic window and target output size being half of input size\n        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))\n\n    .. _Fractional MaxPooling:\n        http://arxiv.org/abs/1412.6071\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "output_size": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "output_ratio": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "_random_samples": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            }
          }
        },
        "gaussian_nll_loss": {
          "Doc": "Gaussian negative log likelihood loss.\n\n    See :class:`~torch.nn.GaussianNLLLoss` for details.\n\n    Args:\n        input: expectation of the Gaussian distribution.\n        target: sample from the Gaussian distribution.\n        var: tensor of positive variance(s), one for each of the expectations\n            in the input (heteroscedastic), or a single one (homoscedastic).\n        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.\n        eps (float, optional): value added to var, for stability. Default: 1e-6.\n        reduction (str, optional): specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the output is the average of all batch member losses,\n            ``'sum'``: the output is the sum of all batch member losses.\n            Default: ``'mean'``.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "var": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "full": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-06"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "gelu": {
          "Doc": "\ngelu(input, approximate = 'none') -> Tensor\n\nWhen the approximate argument is 'none', it applies element-wise the function\n:math:`\\text{GELU}(x) = x * \\Phi(x)`\n\nwhere :math:`\\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.\n\nWhen the approximate argument is 'tanh', Gelu is estimated with\n\n.. math::\n    \\text{GELU}(x) = 0.5 * x * (1 + \\text{Tanh}(\\sqrt(2 / \\pi) * (x + 0.044715 * x^3)))\n\nSee `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "approximate ": {
              "Type": null,
              "Default": "'none'"
            }
          }
        },
        "glu": {
          "Doc": "\n    glu(input, dim=-1) -> Tensor\n\n    The gated linear unit. Computes:\n\n    .. math ::\n        \\text{GLU}(a, b) = a \\otimes \\sigma(b)\n\n    where `input` is split in half along `dim` to form `a` and `b`, :math:`\\sigma`\n    is the sigmoid function and :math:`\\otimes` is the element-wise product between matrices.\n\n    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.\n\n    Args:\n        input (Tensor): input tensor\n        dim (int): dimension on which to split the input. Default: -1\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "dim": {
              "Type": "<class 'int'>",
              "Default": "-1"
            }
          }
        },
        "grid_sample": {
          "Doc": "Given an :attr:`input` and a flow-field :attr:`grid`, computes the\n    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.\n\n    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are\n    supported.\n\n    In the spatial (4-D) case, for :attr:`input` with shape\n    :math:`(N, C, H_\\text{in}, W_\\text{in})` and :attr:`grid` with shape\n    :math:`(N, H_\\text{out}, W_\\text{out}, 2)`, the output will have shape\n    :math:`(N, C, H_\\text{out}, W_\\text{out})`.\n\n    For each output location ``output[n, :, h, w]``, the size-2 vector\n    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,\n    which are used to interpolate the output value ``output[n, :, h, w]``.\n    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the\n    ``x``, ``y``, ``z`` pixel locations for interpolating\n    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or\n    ``bilinear`` interpolation method to sample the input pixels.\n\n    :attr:`grid` specifies the sampling pixel locations normalized by the\n    :attr:`input` spatial dimensions. Therefore, it should have most values in\n    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the\n    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the\n    right-bottom pixel of :attr:`input`.\n\n    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding\n    outputs are handled as defined by :attr:`padding_mode`. Options are\n\n        * ``padding_mode=\"zeros\"``: use ``0`` for out-of-bound grid locations,\n        * ``padding_mode=\"border\"``: use border values for out-of-bound grid locations,\n        * ``padding_mode=\"reflection\"``: use values at locations reflected by\n          the border for out-of-bound grid locations. For location far away\n          from the border, it will keep being reflected until becoming in bound,\n          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``\n          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes\n          ``x'' = -0.5``.\n\n    Note:\n        This function is often used in conjunction with :func:`affine_grid`\n        to build `Spatial Transformer Networks`_ .\n\n    Note:\n        When using the CUDA backend, this operation may induce nondeterministic\n        behaviour in its backward pass that is not easily switched off.\n        Please see the notes on :doc:`/notes/randomness` for background.\n\n    Note:\n        NaN values in :attr:`grid` would be interpreted as ``-1``.\n\n    Args:\n        input (Tensor): input of shape :math:`(N, C, H_\\text{in}, W_\\text{in})` (4-D case)\n                        or :math:`(N, C, D_\\text{in}, H_\\text{in}, W_\\text{in})` (5-D case)\n        grid (Tensor): flow-field of shape :math:`(N, H_\\text{out}, W_\\text{out}, 2)` (4-D case)\n                       or :math:`(N, D_\\text{out}, H_\\text{out}, W_\\text{out}, 3)` (5-D case)\n        mode (str): interpolation mode to calculate output values\n            ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``. Default: ``'bilinear'``\n            Note: ``mode='bicubic'`` supports only 4-D input.\n            When ``mode='bilinear'`` and the input is 5-D, the interpolation mode\n            used internally will actually be trilinear. However, when the input is 4-D,\n            the interpolation mode will legitimately be bilinear.\n        padding_mode (str): padding mode for outside grid values\n            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``\n        align_corners (bool, optional): Geometrically, we consider the pixels of the\n            input  as squares rather than points.\n            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring\n            to the center points of the input's corner pixels. If set to ``False``, they\n            are instead considered as referring to the corner points of the input's corner\n            pixels, making the sampling more resolution agnostic.\n            This option parallels the ``align_corners`` option in\n            :func:`interpolate`, and so whichever option is used here\n            should also be used there to resize the input image before grid sampling.\n            Default: ``False``\n\n    Returns:\n        output (Tensor): output Tensor\n\n    .. _`Spatial Transformer Networks`:\n        https://arxiv.org/abs/1506.02025\n\n    .. warning::\n        When ``align_corners = True``, the grid positions depend on the pixel\n        size relative to the input image size, and so the locations sampled by\n        :func:`grid_sample` will differ for the same input given at different\n        resolutions (that is, after being upsampled or downsampled).\n        The default behavior up to version 1.2.0 was ``align_corners = True``.\n        Since then, the default behavior has been changed to ``align_corners = False``,\n        in order to bring it in line with the default for :func:`interpolate`.\n\n    .. note::\n        ``mode='bicubic'`` is implemented using the `cubic convolution algorithm`_ with :math:`\\alpha=-0.75`.\n        The constant :math:`\\alpha` might be different from packages to packages.\n        For example, `PIL`_ and `OpenCV`_ use -0.5 and -0.75 respectively.\n        This algorithm may \"overshoot\" the range of values it's interpolating.\n        For example, it may produce negative values or values greater than 255 when interpolating input in [0, 255].\n        Clamp the results with :func: `torch.clamp` to ensure they are within the valid range.\n    .. _`cubic convolution algorithm`: https://en.wikipedia.org/wiki/Bicubic_interpolation\n    .. _`PIL`: https://github.com/python-pillow/Pillow/blob/4634eafe3c695a014267eefdce830b4a825beed7/src/libImaging/Resample.c#L51\n    .. _`OpenCV`: https://github.com/opencv/opencv/blob/f345ed564a06178670750bad59526cfa4033be55/modules/imgproc/src/resize.cpp#L908\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "grid": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "mode": {
              "Type": "<class 'str'>",
              "Default": "bilinear"
            },
            "padding_mode": {
              "Type": "<class 'str'>",
              "Default": "zeros"
            },
            "align_corners": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            }
          }
        },
        "group_norm": {
          "Doc": "Applies Group Normalization for last certain number of dimensions.\n\n    See :class:`~torch.nn.GroupNorm` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "num_groups": {
              "Type": "<class 'int'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-05"
            }
          }
        },
        "gumbel_softmax": {
          "Doc": "\n    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.\n\n    Args:\n      logits: `[..., num_features]` unnormalized log probabilities\n      tau: non-negative scalar temperature\n      hard: if ``True``, the returned samples will be discretized as one-hot vectors,\n            but will be differentiated as if it is the soft sample in autograd\n      dim (int): A dimension along which softmax will be computed. Default: -1.\n\n    Returns:\n      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.\n      If ``hard=True``, the returned samples will be one-hot, otherwise they will\n      be probability distributions that sum to 1 across `dim`.\n\n    .. note::\n      This function is here for legacy reasons, may be removed from nn.Functional in the future.\n\n    .. note::\n      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`\n\n      It achieves two things:\n      - makes the output value exactly one-hot\n      (since we add then subtract y_soft value)\n      - makes the gradient equal to y_soft gradient\n      (since we strip all other gradients)\n\n    Examples::\n        >>> logits = torch.randn(20, 32)\n        >>> # Sample soft categorical using reparametrization trick:\n        >>> F.gumbel_softmax(logits, tau=1, hard=False)\n        >>> # Sample hard categorical using \"Straight-through\" trick:\n        >>> F.gumbel_softmax(logits, tau=1, hard=True)\n\n    .. _Link 1:\n        https://arxiv.org/abs/1611.00712\n    .. _Link 2:\n        https://arxiv.org/abs/1611.01144\n    ",
          "Args": {
            "logits": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "tau": {
              "Type": "<class 'float'>",
              "Default": "1"
            },
            "hard": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-10"
            },
            "dim": {
              "Type": "<class 'int'>",
              "Default": "-1"
            }
          }
        },
        "handle_torch_function": {
          "Doc": "Implement a function with checks for ``__torch_function__`` overrides.\n\n    See torch::autograd::handle_torch_function for the equivalent of this\n    function in the C++ implementation.\n\n    Arguments\n    ---------\n    public_api : function\n        Function exposed by the public torch API originally called like\n        ``public_api(*args, **kwargs)`` on which arguments are now being\n        checked.\n    relevant_args : iterable\n        Iterable of arguments to check for __torch_function__ methods.\n    args : tuple\n        Arbitrary positional arguments originally passed into ``public_api``.\n    kwargs : tuple\n        Arbitrary keyword arguments originally passed into ``public_api``.\n\n    Returns\n    -------\n    object\n        Result from calling ``implementation`` or an ``__torch_function__``\n        method, as appropriate.\n\n    Raises\n    ------\n    TypeError : if no implementation is found.\n\n    Example\n    -------\n    >>> def func(a):\n    ...     if has_torch_function_unary(a):\n    ...         return handle_torch_function(func, (a,), a)\n    ...     return a + 0\n    ",
          "Args": {
            "public_api": {
              "Type": "typing.Callable",
              "Default": null
            },
            "relevant_args": {
              "Type": "typing.Iterable[typing.Any]",
              "Default": null
            },
            "*args": {
              "Type": null,
              "Default": null
            },
            "**kwargs": {
              "Type": null,
              "Default": null
            }
          }
        },
        "hardshrink": {
          "Doc": "\nhardshrink(input, lambd=0.5) -> Tensor\n\nApplies the hard shrinkage function element-wise\n\nSee :class:`~torch.nn.Hardshrink` for more details.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "lambd": {
              "Type": null,
              "Default": "0.5"
            }
          }
        },
        "hardsigmoid": {
          "Doc": "Applies the element-wise function\n\n    .. math::\n        \\text{Hardsigmoid}(x) = \\begin{cases}\n            0 & \\text{if~} x \\le -3, \\\\\n            1 & \\text{if~} x \\ge +3, \\\\\n            x / 6 + 1 / 2 & \\text{otherwise}\n        \\end{cases}\n\n    Args:\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n\n    See :class:`~torch.nn.Hardsigmoid` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "hardswish": {
          "Doc": "Applies the hardswish function, element-wise, as described in the paper:\n\n    `Searching for MobileNetV3`_.\n\n    .. math::\n        \\text{Hardswish}(x) = \\begin{cases}\n            0 & \\text{if~} x \\le -3, \\\\\n            x & \\text{if~} x \\ge +3, \\\\\n            x \\cdot (x + 3) /6 & \\text{otherwise}\n        \\end{cases}\n\n    See :class:`~torch.nn.Hardswish` for more details.\n\n    .. _`Searching for MobileNetV3`:\n        https://arxiv.org/abs/1905.02244\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "hardtanh": {
          "Doc": "\n    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor\n\n    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more\n    details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "min_val": {
              "Type": "<class 'float'>",
              "Default": "-1.0"
            },
            "max_val": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "hinge_embedding_loss": {
          "Doc": "hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.HingeEmbeddingLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "huber_loss": {
          "Doc": "Function that uses a squared term if the absolute\n    element-wise error falls below delta and a delta-scaled L1 term otherwise.\n\n    See :class:`~torch.nn.HuberLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "delta": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            }
          }
        },
        "instance_norm": {
          "Doc": "Applies Instance Normalization for each channel in each data sample in a\n    batch.\n\n    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,\n    :class:`~torch.nn.InstanceNorm3d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "running_mean": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "running_var": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "use_input_stats": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "momentum": {
              "Type": "<class 'float'>",
              "Default": "0.1"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-05"
            }
          }
        },
        "interpolate": {
          "Doc": "Down/up samples the input to either the given :attr:`size` or the given\n    :attr:`scale_factor`\n\n    The algorithm used for interpolation is determined by :attr:`mode`.\n\n    Currently temporal, spatial and volumetric sampling are supported, i.e.\n    expected inputs are 3-D, 4-D or 5-D in shape.\n\n    The input dimensions are interpreted in the form:\n    `mini-batch x channels x [optional depth] x [optional height] x width`.\n\n    The modes available for resizing are: `nearest`, `linear` (3D-only),\n    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`, `nearest-exact`\n\n    Args:\n        input (Tensor): the input tensor\n        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):\n            output spatial size.\n        scale_factor (float or Tuple[float]): multiplier for spatial size. If `scale_factor` is a tuple,\n            its length has to match the number of spatial dimensions; `input.dim() - 2`.\n        mode (str): algorithm used for upsampling:\n            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |\n            ``'trilinear'`` | ``'area'`` | ``'nearest-exact'``. Default: ``'nearest'``\n        align_corners (bool, optional): Geometrically, we consider the pixels of the\n            input and output as squares rather than points.\n            If set to ``True``, the input and output tensors are aligned by the\n            center points of their corner pixels, preserving the values at the corner pixels.\n            If set to ``False``, the input and output tensors are aligned by the corner\n            points of their corner pixels, and the interpolation uses edge value padding\n            for out-of-boundary values, making this operation *independent* of input size\n            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`\n            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.\n            Default: ``False``\n        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the\n            interpolation calculation. If `recompute_scale_factor` is ``True``, then\n            `scale_factor` must be passed in and `scale_factor` is used to compute the\n            output `size`. The computed output `size` will be used to infer new scales for\n            the interpolation. Note that when `scale_factor` is floating-point, it may differ\n            from the recomputed `scale_factor` due to rounding and precision issues.\n            If `recompute_scale_factor` is ``False``, then `size` or `scale_factor` will\n            be used directly for interpolation. Default: ``None``.\n        antialias (bool, optional): flag to apply anti-aliasing. Default: ``False``. Using anti-alias\n            option together with ``align_corners=False``, interpolation result would match Pillow\n            result for downsampling operation. Supported modes: ``'bilinear'``, ``'bicubic'``.\n\n    .. note::\n        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce\n        negative values or values greater than 255 for images.\n        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot\n        when displaying the image.\n\n    .. note::\n        Mode ``mode='nearest-exact'`` matches Scikit-Image and PIL nearest neighbours interpolation\n        algorithms and fixes known issues with ``mode='nearest'``. This mode is introduced to keep\n        backward compatibility.\n        Mode ``mode='nearest'`` matches buggy OpenCV's ``INTER_NEAREST`` interpolation algorithm.\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            },
            "scale_factor": {
              "Type": "typing.Optional[typing.List[float]]",
              "Default": "None"
            },
            "mode": {
              "Type": "<class 'str'>",
              "Default": "nearest"
            },
            "align_corners": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "recompute_scale_factor": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "antialias": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "kl_div": {
          "Doc": "The `Kullback-Leibler divergence Loss\n    <https://en.wikipedia.org/wiki/Kullback-Leibler_divergence>`__\n\n    See :class:`~torch.nn.KLDivLoss` for details.\n\n    Args:\n        input: Tensor of arbitrary shape in log-probabilities.\n        target: Tensor of the same shape as input. See :attr:`log_target` for\n            the target's interpretation.\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.\n            ``'none'``: no reduction will be applied\n            ``'batchmean'``: the sum of the output will be divided by the batchsize\n            ``'sum'``: the output will be summed\n            ``'mean'``: the output will be divided by the number of elements in the output\n            Default: ``'mean'``\n        log_target (bool): A flag indicating whether ``target`` is passed in the log space.\n            It is recommended to pass certain distributions (like ``softmax``)\n            in the log space to avoid numerical issues caused by explicit ``log``.\n            Default: ``False``\n\n    .. note::\n        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,\n        and in the meantime, specifying either of those two args will override :attr:`reduction`.\n\n    .. note::\n        :attr:`reduction` = ``'mean'`` doesn't return the true kl divergence value, please use\n        :attr:`reduction` = ``'batchmean'`` which aligns with KL math definition.\n        In the next major release, ``'mean'`` will be changed to be the same as 'batchmean'.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "log_target": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "l1_loss": {
          "Doc": "l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    Function that takes the mean element-wise absolute value difference.\n\n    See :class:`~torch.nn.L1Loss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "layer_norm": {
          "Doc": "Applies Layer Normalization for last certain number of dimensions.\n\n    See :class:`~torch.nn.LayerNorm` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "normalized_shape": {
              "Type": "typing.List[int]",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-05"
            }
          }
        },
        "leaky_relu": {
          "Doc": "\n    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor\n\n    Applies element-wise,\n    :math:`\\text{LeakyReLU}(x) = \\max(0, x) + \\text{negative\\_slope} * \\min(0, x)`\n\n    See :class:`~torch.nn.LeakyReLU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "negative_slope": {
              "Type": "<class 'float'>",
              "Default": "0.01"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "linear": {
          "Doc": "\nlinear(input, weight, bias=None) -> Tensor\n\nApplies a linear transformation to the incoming data: :math:`y = xA^T + b`.\n\nThis operation supports 2-D :attr:`weight` with :ref:`sparse layout<sparse-docs>`\n\n\n.. warning::\n    Sparse support is a beta feature and some layout(s)/dtype/device combinations may not be supported,\n    or may not have autograd support. If you notice missing functionality please\n    open a feature request.\n\nThis operator supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\nShape:\n\n    - Input: :math:`(*, in\\_features)` where `*` means any number of\n      additional dimensions, including none\n    - Weight: :math:`(out\\_features, in\\_features)` or :math:`(in\\_features)`\n    - Bias: :math:`(out\\_features)` or :math:`()`\n    - Output: :math:`(*, out\\_features)` or :math:`(*)`, based on the shape of the weight\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            },
            "bias": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "local_response_norm": {
          "Doc": "Applies local response normalization over an input signal composed of\n    several input planes, where channels occupy the second dimension.\n    Applies normalization across channels.\n\n    See :class:`~torch.nn.LocalResponseNorm` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size": {
              "Type": "<class 'int'>",
              "Default": null
            },
            "alpha": {
              "Type": "<class 'float'>",
              "Default": "0.0001"
            },
            "beta": {
              "Type": "<class 'float'>",
              "Default": "0.75"
            },
            "k": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            }
          }
        },
        "log_softmax": {
          "Doc": "Applies a softmax followed by a logarithm.\n\n    While mathematically equivalent to log(softmax(x)), doing these two\n    operations separately is slower and numerically unstable. This function\n    uses an alternative formulation to compute the output and gradient correctly.\n\n    See :class:`~torch.nn.LogSoftmax` for more details.\n\n    Args:\n        input (Tensor): input\n        dim (int): A dimension along which log_softmax will be computed.\n        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.\n          If specified, the input tensor is cast to :attr:`dtype` before the operation\n          is performed. This is useful for preventing data type overflows. Default: None.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "dim": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            },
            "_stacklevel": {
              "Type": "<class 'int'>",
              "Default": "3"
            },
            "dtype": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            }
          }
        },
        "logsigmoid": {
          "Doc": "\nlogsigmoid(input) -> Tensor\n\nApplies element-wise :math:`\\text{LogSigmoid}(x_i) = \\log \\left(\\frac{1}{1 + \\exp(-x_i)}\\right)`\n\nSee :class:`~torch.nn.LogSigmoid` for more details.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            }
          }
        },
        "lp_pool1d": {
          "Doc": "Applies a 1D power-average pooling over an input signal composed of\n    several input planes. If the sum of all inputs to the power of `p` is\n    zero, the gradient is set to zero as well.\n\n    See :class:`~torch.nn.LPPool1d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "norm_type": {
              "Type": "typing.Union[int, float]",
              "Default": null
            },
            "kernel_size": {
              "Type": "<class 'int'>",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "ceil_mode": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "lp_pool2d": {
          "Doc": "Applies a 2D power-average pooling over an input signal composed of\n    several input planes. If the sum of all inputs to the power of `p` is\n    zero, the gradient is set to zero as well.\n\n    See :class:`~torch.nn.LPPool2d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "norm_type": {
              "Type": "typing.Union[int, float]",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "ceil_mode": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "margin_ranking_loss": {
          "Doc": "margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.MarginRankingLoss` for details.\n    ",
          "Args": {
            "input1": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "input2": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "0"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "max_pool1d_with_indices": {
          "Doc": "\n    max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)\n\n    Applies a 1D max pooling over an input signal composed of several input\n    planes.\n\n    .. note::\n        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from\n        what seen in :class:`~torch.nn.MaxPool1d`, and will change in a future release.\n\n    See :class:`~torch.nn.MaxPool1d` for details.\n\n    Args:\n        input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`, minibatch dim optional.\n        kernel_size: the size of the window. Can be a single number or a\n            tuple `(kW,)`\n        stride: the stride of the window. Can be a single number or a tuple\n            `(sW,)`. Default: :attr:`kernel_size`\n        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.\n        dilation: The stride between elements within a sliding window, must be > 0.\n        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This\n                   ensures that every element in the input tensor is covered by a sliding window.\n        return_indices: If ``True``, will return the argmax along with the max values.\n                        Useful for :class:`torch.nn.functional.max_unpool1d` later\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "dilation": {
              "Type": "None",
              "Default": "1"
            },
            "ceil_mode": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "max_pool2d_with_indices": {
          "Doc": "\n    max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)\n\n    Applies a 2D max pooling over an input signal composed of several input\n    planes.\n\n    .. note::\n        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from\n        what seen in :class:`~torch.nn.MaxPool2d`, and will change in a future release.\n\n    See :class:`~torch.nn.MaxPool2d` for details.\n\n    Args:\n        input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`, minibatch dim optional.\n        kernel_size: size of the pooling region. Can be a single number or a\n            tuple `(kH, kW)`\n        stride: stride of the pooling operation. Can be a single number or a\n            tuple `(sH, sW)`. Default: :attr:`kernel_size`\n        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.\n        dilation: The stride between elements within a sliding window, must be > 0.\n        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This\n                   ensures that every element in the input tensor is covered by a sliding window.\n        return_indices: If ``True``, will return the argmax along with the max values.\n                        Useful for :class:`torch.nn.functional.max_unpool2d` later\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "dilation": {
              "Type": "None",
              "Default": "1"
            },
            "ceil_mode": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "max_pool3d_with_indices": {
          "Doc": "\n    max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)\n\n    Applies a 3D max pooling over an input signal composed of several input\n    planes.\n\n    .. note::\n        The order of :attr:`ceil_mode` and :attr:`return_indices` is different from\n        what seen in :class:`~torch.nn.MaxPool3d`, and will change in a future release.\n\n    See :class:`~torch.nn.MaxPool3d` for details.\n\n    Args:\n        input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iD, iH , iW)`, minibatch dim optional.\n        kernel_size: size of the pooling region. Can be a single number or a\n                     tuple `(kT, kH, kW)`\n        stride: stride of the pooling operation. Can be a single number or a\n                tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`\n        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.\n        dilation: The stride between elements within a sliding window, must be > 0.\n        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This\n                   ensures that every element in the input tensor is covered by a sliding window.\n        return_indices: If ``True``, will return the argmax along with the max values.\n                        Useful for :class:`torch.nn.functional.max_unpool3d` later\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "dilation": {
              "Type": "None",
              "Default": "1"
            },
            "ceil_mode": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "return_indices": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "max_unpool1d": {
          "Doc": "Computes a partial inverse of :class:`MaxPool1d`.\n\n    See :class:`~torch.nn.MaxUnpool1d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "indices": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "output_size": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            }
          }
        },
        "max_unpool2d": {
          "Doc": "Computes a partial inverse of :class:`MaxPool2d`.\n\n    See :class:`~torch.nn.MaxUnpool2d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "indices": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "output_size": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            }
          }
        },
        "max_unpool3d": {
          "Doc": "Computes a partial inverse of :class:`MaxPool3d`.\n\n    See :class:`~torch.nn.MaxUnpool3d` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "indices": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "stride": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "output_size": {
              "Type": "<class 'NoneType'>",
              "Default": "None"
            }
          }
        },
        "mish": {
          "Doc": "Applies the Mish function, element-wise.\n    Mish: A Self Regularized Non-Monotonic Neural Activation Function.\n\n    .. math::\n        \\text{Mish}(x) = x * \\text{Tanh}(\\text{Softplus}(x))\n\n    .. note::\n        See `Mish: A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_\n\n    See :class:`~torch.nn.Mish` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "mse_loss": {
          "Doc": "mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    Measures the element-wise mean squared error.\n\n    See :class:`~torch.nn.MSELoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "multi_head_attention_forward": {
          "Doc": "\n    Args:\n        query, key, value: map a query and a set of key-value pairs to an output.\n            See \"Attention Is All You Need\" for more details.\n        embed_dim_to_check: total dimension of the model.\n        num_heads: parallel attention heads.\n        in_proj_weight, in_proj_bias: input projection weight and bias.\n        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.\n        add_zero_attn: add a new batch of zeros to the key and\n                       value sequences at dim=1.\n        dropout_p: probability of an element to be zeroed.\n        out_proj_weight, out_proj_bias: the output projection weight and bias.\n        training: apply dropout if is ``True``.\n        key_padding_mask: if provided, specified padding elements in the key will\n            be ignored by the attention. This is an binary mask. When the value is True,\n            the corresponding value on the attention layer will be filled with -inf.\n        need_weights: output attn_output_weights.\n        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all\n            the batches while a 3D mask allows to specify a different mask for the entries of each batch.\n        is_causal: If specified, applies a causal mask as attention mask, and ignores\n            attn_mask for computing scaled dot product attention.\n            Default: ``False``.\n        use_separate_proj_weight: the function accept the proj. weights for query, key,\n            and value in different forms. If false, in_proj_weight will be used, which is\n            a combination of q_proj_weight, k_proj_weight, v_proj_weight.\n        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.\n        static_k, static_v: static key and value used for attention operators.\n        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.\n            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect\n            when ``need_weights=True.``. Default: True\n\n\n    Shape:\n        Inputs:\n        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is\n          the embedding dimension.\n        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is\n          the embedding dimension.\n        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is\n          the embedding dimension.\n        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.\n          If a FloatTensor is provided, it will be directly added to the value.\n          If a BoolTensor is provided, the positions with the\n          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.\n        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.\n          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,\n          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked\n          positions. If a BoolTensor is provided, positions with ``True``\n          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor\n          is provided, it will be added to the attention weight.\n        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,\n          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.\n        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,\n          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.\n\n        Outputs:\n        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,\n          E is the embedding dimension.\n        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns\n          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or\n          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and\n          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per\n          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.\n    ",
          "Args": {
            "query": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "key": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "value": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "embed_dim_to_check": {
              "Type": "<class 'int'>",
              "Default": null
            },
            "num_heads": {
              "Type": "<class 'int'>",
              "Default": null
            },
            "in_proj_weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "in_proj_bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "bias_k": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "bias_v": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "add_zero_attn": {
              "Type": "<class 'bool'>",
              "Default": null
            },
            "dropout_p": {
              "Type": "<class 'float'>",
              "Default": null
            },
            "out_proj_weight": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "out_proj_bias": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": null
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "key_padding_mask": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "need_weights": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "attn_mask": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "use_separate_proj_weight": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "q_proj_weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "k_proj_weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "v_proj_weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "static_k": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "static_v": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "average_attn_weights": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "is_causal": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "multi_margin_loss": {
          "Doc": "multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.MultiMarginLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'int'>",
              "Default": "1"
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "multilabel_margin_loss": {
          "Doc": "multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.MultiLabelMarginLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "multilabel_soft_margin_loss": {
          "Doc": "multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "native_channel_shuffle": {
          "Doc": "\nnative_channel_shuffle(input, groups) -> Tensor\n\nNative kernel level implementation of the `channel_shuffle`.\nThis function might become private in future releases, use with caution.\n\nDivide the channels in a tensor of shape :math:`(*, C , H, W)`\ninto g groups and rearrange them as :math:`(*, C \\frac g, g, H, W)`,\nwhile keeping the original tensor shape.\n\nSee :class:`~torch.nn.ChannelShuffle` for details.\n\nArgs:\n    input (Tensor): the input tensor\n    groups (int): number of groups to divide channels in and rearrange.\n\nExamples::\n\n    >>> input = torch.randn(1, 4, 2, 2)\n    >>> print(input)\n    [[[[1, 2],\n       [3, 4]],\n      [[5, 6],\n       [7, 8]],\n      [[9, 10],\n       [11, 12]],\n      [[13, 14],\n       [15, 16]],\n     ]]\n    >>> output = torch.nn.functional.native_channel_shuffle(input, 2)\n    >>> print(output)\n    [[[[1, 2],\n       [3, 4]],\n      [[9, 10],\n       [11, 12]],\n      [[5, 6],\n       [7, 8]],\n      [[13, 14],\n       [15, 16]],\n     ]]\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "groups": {
              "Type": null,
              "Default": null
            }
          }
        },
        "nll_loss": {
          "Doc": "The negative log likelihood loss.\n\n    See :class:`~torch.nn.NLLLoss` for details.\n\n    Args:\n        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`\n            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \\geq 1`\n            in the case of K-dimensional loss. `input` is expected to be log-probabilities.\n        target: :math:`(N)` where each value is :math:`0 \\leq \\text{targets}[i] \\leq C-1`,\n            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \\geq 1` for\n            K-dimensional loss.\n        weight (Tensor, optional): a manual rescaling weight given to each\n            class. If given, has to be a Tensor of size `C`\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        ignore_index (int, optional): Specifies a target value that is ignored\n            and does not contribute to the input gradient. When :attr:`size_average` is\n            ``True``, the loss is averaged over non-ignored targets. Default: -100\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n    Example::\n\n        >>> # input is of size N x C = 3 x 5\n        >>> input = torch.randn(3, 5, requires_grad=True)\n        >>> # each element in target has to have 0 <= value < C\n        >>> target = torch.tensor([1, 0, 4])\n        >>> output = F.nll_loss(F.log_softmax(input, dim=1), target)\n        >>> output.backward()\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "weight": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "ignore_index": {
              "Type": "<class 'int'>",
              "Default": "-100"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "normalize": {
          "Doc": "Performs :math:`L_p` normalization of inputs over specified dimension.\n\n    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each\n    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as\n\n    .. math::\n        v = \\frac{v}{\\max(\\lVert v \\rVert_p, \\epsilon)}.\n\n    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.\n\n    Args:\n        input: input tensor of any shape\n        p (float): the exponent value in the norm formulation. Default: 2\n        dim (int): the dimension to reduce. Default: 1\n        eps (float): small value to avoid division by zero. Default: 1e-12\n        out (Tensor, optional): the output tensor. If :attr:`out` is used, this\n                                operation won't be differentiable.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "2.0"
            },
            "dim": {
              "Type": "<class 'int'>",
              "Default": "1"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-12"
            },
            "out": {
              "Type": "typing.Optional[torch.Tensor]",
              "Default": "None"
            }
          }
        },
        "one_hot": {
          "Doc": "\none_hot(tensor, num_classes=-1) -> LongTensor\n\nTakes LongTensor with index values of shape ``(*)`` and returns a tensor\nof shape ``(*, num_classes)`` that have zeros everywhere except where the\nindex of last dimension matches the corresponding value of the input tensor,\nin which case it will be 1.\n\nSee also `One-hot on Wikipedia`_ .\n\n.. _One-hot on Wikipedia:\n    https://en.wikipedia.org/wiki/One-hot\n\nArguments:\n    tensor (LongTensor): class values of any shape.\n    num_classes (int):  Total number of classes. If set to -1, the number\n        of classes will be inferred as one greater than the largest class\n        value in the input tensor.\n\nReturns:\n    LongTensor that has one more dimension with 1 values at the\n    index of last dimension indicated by the input, and 0 everywhere\n    else.\n\nExamples:\n    >>> F.one_hot(torch.arange(0, 5) % 3)\n    tensor([[1, 0, 0],\n            [0, 1, 0],\n            [0, 0, 1],\n            [1, 0, 0],\n            [0, 1, 0]])\n    >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)\n    tensor([[1, 0, 0, 0, 0],\n            [0, 1, 0, 0, 0],\n            [0, 0, 1, 0, 0],\n            [1, 0, 0, 0, 0],\n            [0, 1, 0, 0, 0]])\n    >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)\n    tensor([[[1, 0, 0],\n             [0, 1, 0]],\n            [[0, 0, 1],\n             [1, 0, 0]],\n            [[0, 1, 0],\n             [0, 0, 1]]])\n",
          "Args": {
            "tensor": {
              "Type": null,
              "Default": null
            },
            "num_classes": {
              "Type": null,
              "Default": "-1"
            }
          }
        },
        "pad": {
          "Doc": "\npad(input, pad, mode=\"constant\", value=None) -> Tensor\n\nPads tensor.\n\nPadding size:\n    The padding size by which to pad some dimensions of :attr:`input`\n    are described starting from the last dimension and moving forward.\n    :math:`\\left\\lfloor\\frac{\\text{len(pad)}}{2}\\right\\rfloor` dimensions\n    of ``input`` will be padded.\n    For example, to pad only the last dimension of the input tensor, then\n    :attr:`pad` has the form\n    :math:`(\\text{padding\\_left}, \\text{padding\\_right})`;\n    to pad the last 2 dimensions of the input tensor, then use\n    :math:`(\\text{padding\\_left}, \\text{padding\\_right},`\n    :math:`\\text{padding\\_top}, \\text{padding\\_bottom})`;\n    to pad the last 3 dimensions, use\n    :math:`(\\text{padding\\_left}, \\text{padding\\_right},`\n    :math:`\\text{padding\\_top}, \\text{padding\\_bottom}`\n    :math:`\\text{padding\\_front}, \\text{padding\\_back})`.\n\nPadding mode:\n    See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and\n    :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the\n    padding modes works. Constant padding is implemented for arbitrary dimensions.\n    Replicate and reflection padding are implemented for padding the last 3\n    dimensions of a 4D or 5D input tensor, the last 2 dimensions of a 3D\n    or 4D input tensor, or the last dimension of a 2D or 3D input tensor.\n\nNote:\n    When using the CUDA backend, this operation may induce nondeterministic\n    behaviour in its backward pass that is not easily switched off.\n    Please see the notes on :doc:`/notes/randomness` for background.\n\nArgs:\n    input (Tensor): N-dimensional tensor\n    pad (tuple): m-elements tuple, where\n        :math:`\\frac{m}{2} \\leq` input dimensions and :math:`m` is even.\n    mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.\n        Default: ``'constant'``\n    value: fill value for ``'constant'`` padding. Default: ``0``\n\nExamples::\n\n    >>> t4d = torch.empty(3, 3, 4, 2)\n    >>> p1d = (1, 1) # pad last dim by 1 on each side\n    >>> out = F.pad(t4d, p1d, \"constant\", 0)  # effectively zero padding\n    >>> print(out.size())\n    torch.Size([3, 3, 4, 4])\n    >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)\n    >>> out = F.pad(t4d, p2d, \"constant\", 0)\n    >>> print(out.size())\n    torch.Size([3, 3, 8, 4])\n    >>> t4d = torch.empty(3, 3, 4, 2)\n    >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)\n    >>> out = F.pad(t4d, p3d, \"constant\", 0)\n    >>> print(out.size())\n    torch.Size([3, 9, 7, 3])\n\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "pad": {
              "Type": null,
              "Default": null
            },
            "mode": {
              "Type": null,
              "Default": "\"constant\""
            },
            "value": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "pairwise_distance": {
          "Doc": "\npairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False) -> Tensor\n\nSee :class:`torch.nn.PairwiseDistance` for details\n",
          "Args": {
            "x1": {
              "Type": null,
              "Default": null
            },
            "x2": {
              "Type": null,
              "Default": null
            },
            "p": {
              "Type": null,
              "Default": "2.0"
            },
            "eps": {
              "Type": null,
              "Default": "1e-6"
            },
            "keepdim": {
              "Type": null,
              "Default": "False"
            }
          }
        },
        "pdist": {
          "Doc": "\npdist(input, p=2) -> Tensor\n\nComputes the p-norm distance between every pair of row vectors in the input.\nThis is identical to the upper triangular portion, excluding the diagonal, of\n`torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster\nif the rows are contiguous.\n\nIf input has shape :math:`N \\times M` then the output will have shape\n:math:`\\frac{1}{2} N (N - 1)`.\n\nThis function is equivalent to ``scipy.spatial.distance.pdist(input,\n'minkowski', p=p)`` if :math:`p \\in (0, \\infty)`. When :math:`p = 0` it is\nequivalent to ``scipy.spatial.distance.pdist(input, 'hamming') * M``.\nWhen :math:`p = \\infty`, the closest scipy function is\n``scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())``.\n\nArgs:\n    input: input tensor of shape :math:`N \\times M`.\n    p: p value for the p-norm distance to calculate between each vector pair\n        :math:`\\in [0, \\infty]`.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "p": {
              "Type": null,
              "Default": "2"
            }
          }
        },
        "pixel_shuffle": {
          "Doc": "\npixel_shuffle(input, upscale_factor) -> Tensor\n\nRearranges elements in a tensor of shape :math:`(*, C \\times r^2, H, W)` to a\ntensor of shape :math:`(*, C, H \\times r, W \\times r)`, where r is the :attr:`upscale_factor`.\n\nSee :class:`~torch.nn.PixelShuffle` for details.\n\nArgs:\n    input (Tensor): the input tensor\n    upscale_factor (int): factor to increase spatial resolution by\n\nExamples::\n\n    >>> input = torch.randn(1, 9, 4, 4)\n    >>> output = torch.nn.functional.pixel_shuffle(input, 3)\n    >>> print(output.size())\n    torch.Size([1, 1, 12, 12])\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "upscale_factor": {
              "Type": null,
              "Default": null
            }
          }
        },
        "pixel_unshuffle": {
          "Doc": "\npixel_unshuffle(input, downscale_factor) -> Tensor\n\nReverses the :class:`~torch.nn.PixelShuffle` operation by rearranging elements in a\ntensor of shape :math:`(*, C, H \\times r, W \\times r)` to a tensor of shape\n:math:`(*, C \\times r^2, H, W)`, where r is the :attr:`downscale_factor`.\n\nSee :class:`~torch.nn.PixelUnshuffle` for details.\n\nArgs:\n    input (Tensor): the input tensor\n    downscale_factor (int): factor to increase spatial resolution by\n\nExamples::\n\n    >>> input = torch.randn(1, 1, 12, 12)\n    >>> output = torch.nn.functional.pixel_unshuffle(input, 3)\n    >>> print(output.size())\n    torch.Size([1, 9, 4, 4])\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "downscale_factor": {
              "Type": null,
              "Default": null
            }
          }
        },
        "poisson_nll_loss": {
          "Doc": "Poisson negative log likelihood loss.\n\n    See :class:`~torch.nn.PoissonNLLLoss` for details.\n\n    Args:\n        input: expectation of underlying Poisson distribution.\n        target: random sample :math:`target \\sim \\text{Poisson}(input)`.\n        log_input: if ``True`` the loss is computed as\n            :math:`\\exp(\\text{input}) - \\text{target} * \\text{input}`, if ``False`` then loss is\n            :math:`\\text{input} - \\text{target} * \\log(\\text{input}+\\text{eps})`. Default: ``True``\n        full: whether to compute full loss, i. e. to add the Stirling\n            approximation term. Default: ``False``\n            :math:`\\text{target} * \\log(\\text{target}) - \\text{target} + 0.5 * \\log(2 * \\pi * \\text{target})`.\n        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,\n            the losses are averaged over each loss element in the batch. Note that for\n            some losses, there multiple elements per sample. If the field :attr:`size_average`\n            is set to ``False``, the losses are instead summed for each minibatch. Ignored\n            when reduce is ``False``. Default: ``True``\n        eps (float, optional): Small value to avoid evaluation of :math:`\\log(0)` when\n            :attr:`log_input`\\ =\\ ``False``. Default: 1e-8\n        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the\n            losses are averaged or summed over observations for each minibatch depending\n            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per\n            batch element instead and ignores :attr:`size_average`. Default: ``True``\n        reduction (str, optional): Specifies the reduction to apply to the output:\n            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,\n            ``'mean'``: the sum of the output will be divided by the number of\n            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`\n            and :attr:`reduce` are in the process of being deprecated, and in the meantime,\n            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``\n\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "log_input": {
              "Type": "<class 'bool'>",
              "Default": "True"
            },
            "full": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-08"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "prelu": {
          "Doc": "prelu(input, weight) -> Tensor\n\nApplies element-wise the function\n:math:`\\text{PReLU}(x) = \\max(0,x) + \\text{weight} * \\min(0,x)` where weight is a\nlearnable parameter.\n\n.. note::\n    `weight` is expected to be a scalar or 1-D tensor. If `weight` is 1-D,\n    its size must match the number of input channels, determined by\n    `input.size(1)` when `input.dim() >= 2`, otherwise 1.\n    In the 1-D case, note that when `input` has dim > 2, `weight` can be expanded\n    to the shape of `input` in a way that is not possible using normal\n    :ref:`broadcasting semantics<broadcasting-semantics>`.\n\nSee :class:`~torch.nn.PReLU` for more details.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "weight": {
              "Type": null,
              "Default": null
            }
          }
        },
        "relu": {
          "Doc": "relu(input, inplace=False) -> Tensor\n\n    Applies the rectified linear unit function element-wise. See\n    :class:`~torch.nn.ReLU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "relu6": {
          "Doc": "relu6(input, inplace=False) -> Tensor\n\n    Applies the element-wise function :math:`\\text{ReLU6}(x) = \\min(\\max(0,x), 6)`.\n\n    See :class:`~torch.nn.ReLU6` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "rrelu": {
          "Doc": "rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor\n\n    Randomized leaky ReLU.\n\n    See :class:`~torch.nn.RReLU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "lower": {
              "Type": "<class 'float'>",
              "Default": "0.125"
            },
            "upper": {
              "Type": "<class 'float'>",
              "Default": "0.3333333333333333"
            },
            "training": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "scaled_dot_product_attention": {
          "Doc": "\nscaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False) -> Tensor:\n\nComputes scaled dot product attention on query, key and value tensors, using\nan optional attention mask if passed, and applying dropout if a probability\ngreater than 0.0 is specified.\n\n.. code-block:: python\n\n    # Efficient implementation equivalent to the following:\n    attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask\n    attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask\n    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)\n    attn_weight = torch.dropout(attn_weight, dropout_p)\n    return attn_weight @ V\n\n.. warning:: This function is beta and subject to change.\n\nNote:\n\n    There are currently three supported implementations of scaled dot product attention:\n\n        - `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_\n        - `Memory-Efficient Attention`_\n        - A PyTorch implementation defined in C++ matching the above formulation\n\n    The function may call optimized kernels for improved performance when using the CUDA backend.\n    For all other backends, the PyTorch implementation will be used.\n\n    All implementations are enabled by default. Scaled dot product attention attempts to automatically select the\n    most optimal implementation based on the inputs. In order to provide more fine-grained control over what implementation\n    is used, the following functions are provided for enabling and disabling implementations.\n    The context manager is the preferred mechanism:\n\n        - :func:`torch.backends.cuda.sdp_kernel`: A context manager used to enable/disable any of the implementations.\n        - :func:`torch.backends.cuda.enable_flash_sdp`: Enables or Disables FlashAttention.\n        - :func:`torch.backends.cuda.enable_mem_efficient_sdp`: Enables or Disables Memory-Efficient Attention.\n        - :func:`torch.backends.cuda.enable_math_sdp`: Enables or Disables the PyTorch C++ implementation.\n\n    Each of the fused kernels has specific input limitations. If the user requires the use of a specific fused implementation,\n    disable the PyTorch C++ implementation using :func:`torch.backends.cuda.sdp_kernel`.\n    In the event that a fused implementation is not available, an error will be raised with the\n    reasons why the fused implementation cannot run.\n\n    Due to the nature of fusing floating point operations, the output of this function may be different\n    depending on what backend kernel is chosen.\n    The c++ implementation supports torch.float64 and can be used when higher precision is required.\n    For more information please see :doc:`/notes/numerical_accuracy`\n\nNote:\n    In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n\nArgs:\n    query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.\n    key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.\n    value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.\n    attn_mask (optional Tensor): Attention mask; shape :math:`(N, ..., L, S)`. Two types of masks are supported.\n        A boolean mask where a value of True indicates that the element *should* take part in attention.\n        A float mask of the same type as query, key, value that is added to the attention score.\n    dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied\n    is_causal (bool): If true, assumes causal attention masking and errors if both attn_mask and is_causal\n        are set.\n\n\nReturns:\n    output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.\n\nShape legend:\n    - :math:`N: \\text{Batch size} ... : \\text{Any number of other batch dimensions (optional)}`\n    - :math:`S: \\text{Source sequence length}`\n    - :math:`L: \\text{Target sequence length}`\n    - :math:`E: \\text{Embedding dimension of the query and key}`\n    - :math:`Ev: \\text{Embedding dimension of the value}`\n\nExamples::\n\n    >>> # Optionally use the context manager to ensure one of the fused kerenels is run\n    >>> query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=\"cuda\")\n    >>> key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=\"cuda\")\n    >>> value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=\"cuda\")\n    >>> with torch.backends.cuda.sdp_kernel(enable_math=False):\n    >>>     F.scaled_dot_product_attention(query,key,value)\n\n.. _FlashAttention\\: Fast and Memory-Efficient Exact Attention with IO-Awareness:\n    https://arxiv.org/abs/2205.14135\n.. _Memory-Efficient Attention:\n    https://github.com/facebookresearch/xformers\n\n",
          "Args": {
            "query": {
              "Type": null,
              "Default": null
            },
            "key": {
              "Type": null,
              "Default": null
            },
            "value": {
              "Type": null,
              "Default": null
            },
            "attn_mask": {
              "Type": null,
              "Default": "None"
            },
            "dropout_p": {
              "Type": null,
              "Default": "0.0"
            },
            "is_causal": {
              "Type": null,
              "Default": "False"
            }
          }
        },
        "selu": {
          "Doc": "selu(input, inplace=False) -> Tensor\n\n    Applies element-wise,\n    :math:`\\text{SELU}(x) = scale * (\\max(0,x) + \\min(0, \\alpha * (\\exp(x) - 1)))`,\n    with :math:`\\alpha=1.6732632423543772848170429916717` and\n    :math:`scale=1.0507009873554804934193349852946`.\n\n    See :class:`~torch.nn.SELU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "sigmoid": {
          "Doc": "sigmoid(input) -> Tensor\n\n    Applies the element-wise function :math:`\\text{Sigmoid}(x) = \\frac{1}{1 + \\exp(-x)}`\n\n    See :class:`~torch.nn.Sigmoid` for more details.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            }
          }
        },
        "silu": {
          "Doc": "Applies the Sigmoid Linear Unit (SiLU) function, element-wise.\n    The SiLU function is also known as the swish function.\n\n    .. math::\n        \\text{silu}(x) = x * \\sigma(x), \\text{where } \\sigma(x) \\text{ is the logistic sigmoid.}\n\n    .. note::\n        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_\n        where the SiLU (Sigmoid Linear Unit) was originally coined, and see\n        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation\n        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:\n        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_\n        where the SiLU was experimented with later.\n\n    See :class:`~torch.nn.SiLU` for more details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "inplace": {
              "Type": "<class 'bool'>",
              "Default": "False"
            }
          }
        },
        "smooth_l1_loss": {
          "Doc": "Function that uses a squared term if the absolute\n    element-wise error falls below beta and an L1 term otherwise.\n\n    See :class:`~torch.nn.SmoothL1Loss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            },
            "beta": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            }
          }
        },
        "soft_margin_loss": {
          "Doc": "soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor\n\n    See :class:`~torch.nn.SoftMarginLoss` for details.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "target": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "softmax": {
          "Doc": "Applies a softmax function.\n\n    Softmax is defined as:\n\n    :math:`\\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}`\n\n    It is applied to all slices along dim, and will re-scale them so that the elements\n    lie in the range `[0, 1]` and sum to 1.\n\n    See :class:`~torch.nn.Softmax` for more details.\n\n    Args:\n        input (Tensor): input\n        dim (int): A dimension along which softmax will be computed.\n        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.\n          If specified, the input tensor is casted to :attr:`dtype` before the operation\n          is performed. This is useful for preventing data type overflows. Default: None.\n\n    .. note::\n        This function doesn't work directly with NLLLoss,\n        which expects the Log to be computed between the Softmax and itself.\n        Use log_softmax instead (it's faster and has better numerical properties).\n\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "dim": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            },
            "_stacklevel": {
              "Type": "<class 'int'>",
              "Default": "3"
            },
            "dtype": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            }
          }
        },
        "softmin": {
          "Doc": "Applies a softmin function.\n\n    Note that :math:`\\text{Softmin}(x) = \\text{Softmax}(-x)`. See softmax definition for mathematical formula.\n\n    See :class:`~torch.nn.Softmin` for more details.\n\n    Args:\n        input (Tensor): input\n        dim (int): A dimension along which softmin will be computed (so every slice\n            along dim will sum to 1).\n        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.\n          If specified, the input tensor is casted to :attr:`dtype` before the operation\n          is performed. This is useful for preventing data type overflows. Default: None.\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "dim": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            },
            "_stacklevel": {
              "Type": "<class 'int'>",
              "Default": "3"
            },
            "dtype": {
              "Type": "typing.Optional[int]",
              "Default": "None"
            }
          }
        },
        "softplus": {
          "Doc": "\nsoftplus(input, beta=1, threshold=20) -> Tensor\n\nApplies element-wise, the function :math:`\\text{Softplus}(x) = \\frac{1}{\\beta} * \\log(1 + \\exp(\\beta * x))`.\n\nFor numerical stability the implementation reverts to the linear function\nwhen :math:`input \\times \\beta > threshold`.\n\nSee :class:`~torch.nn.Softplus` for more details.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "beta": {
              "Type": null,
              "Default": "1"
            },
            "threshold": {
              "Type": null,
              "Default": "20"
            }
          }
        },
        "softshrink": {
          "Doc": "\nsoftshrink(input, lambd=0.5) -> Tensor\n\nApplies the soft shrinkage function elementwise\n\nSee :class:`~torch.nn.Softshrink` for more details.\n",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "lambd": {
              "Type": null,
              "Default": "0.5"
            }
          }
        },
        "softsign": {
          "Doc": "softsign(input) -> Tensor\n\n    Applies element-wise, the function :math:`\\text{SoftSign}(x) = \\frac{x}{1 + |x|}`\n\n    See :class:`~torch.nn.Softsign` for more details.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            }
          }
        },
        "tanh": {
          "Doc": "tanh(input) -> Tensor\n\n    Applies element-wise,\n    :math:`\\text{Tanh}(x) = \\tanh(x) = \\frac{\\exp(x) - \\exp(-x)}{\\exp(x) + \\exp(-x)}`\n\n    See :class:`~torch.nn.Tanh` for more details.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            }
          }
        },
        "tanhshrink": {
          "Doc": "tanhshrink(input) -> Tensor\n\n    Applies element-wise, :math:`\\text{Tanhshrink}(x) = x - \\text{Tanh}(x)`\n\n    See :class:`~torch.nn.Tanhshrink` for more details.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            }
          }
        },
        "triplet_margin_loss": {
          "Doc": "\n    See :class:`~torch.nn.TripletMarginLoss` for details\n    ",
          "Args": {
            "anchor": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "positive": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "negative": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "p": {
              "Type": "<class 'float'>",
              "Default": "2"
            },
            "eps": {
              "Type": "<class 'float'>",
              "Default": "1e-06"
            },
            "swap": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "size_average": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduce": {
              "Type": "typing.Optional[bool]",
              "Default": "None"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "triplet_margin_with_distance_loss": {
          "Doc": "\n    See :class:`~torch.nn.TripletMarginWithDistanceLoss` for details.\n    ",
          "Args": {
            "anchor": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "positive": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "negative": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "distance_function": {
              "Type": "typing.Optional[typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]",
              "Default": "None"
            },
            "margin": {
              "Type": "<class 'float'>",
              "Default": "1.0"
            },
            "swap": {
              "Type": "<class 'bool'>",
              "Default": "False"
            },
            "reduction": {
              "Type": "<class 'str'>",
              "Default": "mean"
            }
          }
        },
        "unfold": {
          "Doc": "Extracts sliding local blocks from a batched input tensor.\n\n    .. warning::\n        Currently, only 4-D input tensors (batched image-like tensors) are\n        supported.\n\n    .. warning::\n\n        More than one element of the unfolded tensor may refer to a single\n        memory location. As a result, in-place operations (especially ones that\n        are vectorized) may result in incorrect behavior. If you need to write\n        to the tensor, please clone it first.\n\n\n    See :class:`torch.nn.Unfold` for details\n    ",
          "Args": {
            "input": {
              "Type": "<class 'torch.Tensor'>",
              "Default": null
            },
            "kernel_size": {
              "Type": "None",
              "Default": null
            },
            "dilation": {
              "Type": "None",
              "Default": "1"
            },
            "padding": {
              "Type": "None",
              "Default": "0"
            },
            "stride": {
              "Type": "None",
              "Default": "1"
            }
          }
        },
        "upsample": {
          "Doc": "Upsamples the input to either the given :attr:`size` or the given\n    :attr:`scale_factor`\n\n    .. warning::\n        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.\n        This is equivalent with ``nn.functional.interpolate(...)``.\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n\n    The algorithm used for upsampling is determined by :attr:`mode`.\n\n    Currently temporal, spatial and volumetric upsampling are supported, i.e.\n    expected inputs are 3-D, 4-D or 5-D in shape.\n\n    The input dimensions are interpreted in the form:\n    `mini-batch x channels x [optional depth] x [optional height] x width`.\n\n    The modes available for upsampling are: `nearest`, `linear` (3D-only),\n    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)\n\n    Args:\n        input (Tensor): the input tensor\n        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):\n            output spatial size.\n        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.\n        mode (str): algorithm used for upsampling:\n            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |\n            ``'trilinear'``. Default: ``'nearest'``\n        align_corners (bool, optional): Geometrically, we consider the pixels of the\n            input and output as squares rather than points.\n            If set to ``True``, the input and output tensors are aligned by the\n            center points of their corner pixels, preserving the values at the corner pixels.\n            If set to ``False``, the input and output tensors are aligned by the corner\n            points of their corner pixels, and the interpolation uses edge value padding\n            for out-of-boundary values, making this operation *independent* of input size\n            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`\n            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.\n            Default: ``False``\n\n    .. note::\n        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce\n        negative values or values greater than 255 for images.\n        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot\n        when displaying the image.\n\n    .. warning::\n        With ``align_corners = True``, the linearly interpolating modes\n        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the\n        output and input pixels, and thus the output values can depend on the\n        input size. This was the default behavior for these modes up to version\n        0.3.1. Since then, the default behavior is ``align_corners = False``.\n        See :class:`~torch.nn.Upsample` for concrete examples on how this\n        affects the outputs.\n\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "size": {
              "Type": null,
              "Default": "None"
            },
            "scale_factor": {
              "Type": null,
              "Default": "None"
            },
            "mode": {
              "Type": null,
              "Default": "nearest"
            },
            "align_corners": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "upsample_bilinear": {
          "Doc": "Upsamples the input, using bilinear upsampling.\n\n    .. warning::\n        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.\n        This is equivalent with\n        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.\n\n    Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo\n    volumetric (5 dimensional) inputs.\n\n    Args:\n        input (Tensor): input\n        size (int or Tuple[int, int]): output spatial size.\n        scale_factor (int or Tuple[int, int]): multiplier for spatial size\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "size": {
              "Type": null,
              "Default": "None"
            },
            "scale_factor": {
              "Type": null,
              "Default": "None"
            }
          }
        },
        "upsample_nearest": {
          "Doc": "Upsamples the input, using nearest neighbours' pixel values.\n\n    .. warning::\n        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.\n        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.\n\n    Currently spatial and volumetric upsampling are supported (i.e. expected\n    inputs are 4 or 5 dimensional).\n\n    Args:\n        input (Tensor): input\n        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia\n            size.\n        scale_factor (int): multiplier for spatial size. Has to be an integer.\n\n    Note:\n        This operation may produce nondeterministic gradients when given tensors on a CUDA device. See :doc:`/notes/randomness` for more information.\n    ",
          "Args": {
            "input": {
              "Type": null,
              "Default": null
            },
            "size": {
              "Type": null,
              "Default": "None"
            },
            "scale_factor": {
              "Type": null,
              "Default": "None"
            }
          }
        }
    },
    "Modules": {
      "activation": {
        "Doc": null,
        "Classes": {
          "ReLU": {
            "Doc": "Applies the rectified linear unit function element-wise:\n\n    :math:`\\text{ReLU}(x) = (x)^+ = \\max(0, x)`\n\n    Args:\n        inplace: can optionally do the operation in-place. Default: ``False``\n\n    Shape:\n        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.\n        - Output: :math:`(*)`, same shape as the input.\n\n    .. image:: ../scripts/activation_images/ReLU.png\n\n    Examples::\n\n        >>> m = nn.ReLU()\n        >>> input = torch.randn(2)\n        >>> output = m(input)\n\n\n      An implementation of CReLU - https://arxiv.org/abs/1603.05201\n\n        >>> m = nn.ReLU()\n        >>> input = torch.randn(2).unsqueeze(0)\n        >>> output = torch.cat((m(input),m(-input)))\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "inplace": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "Sigmoid": {
            "Doc": "Applies the element-wise function:\n\n    .. math::\n        \\text{Sigmoid}(x) = \\sigma(x) = \\frac{1}{1 + \\exp(-x)}\n\n\n    Shape:\n        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.\n        - Output: :math:`(*)`, same shape as the input.\n\n    .. image:: ../scripts/activation_images/Sigmoid.png\n\n    Examples::\n\n        >>> m = nn.Sigmoid()\n        >>> input = torch.randn(2)\n        >>> output = m(input)\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "Softmax": {
            "Doc": "Applies the Softmax function to an n-dimensional input Tensor\n    rescaling them so that the elements of the n-dimensional output Tensor\n    lie in the range [0,1] and sum to 1.\n\n    Softmax is defined as:\n\n    .. math::\n        \\text{Softmax}(x_{i}) = \\frac{\\exp(x_i)}{\\sum_j \\exp(x_j)}\n\n    When the input Tensor is a sparse tensor then the unspecifed\n    values are treated as ``-inf``.\n\n    Shape:\n        - Input: :math:`(*)` where `*` means, any number of additional\n          dimensions\n        - Output: :math:`(*)`, same shape as the input\n\n    Returns:\n        a Tensor of the same dimension and shape as the input with\n        values in the range [0, 1]\n\n    Args:\n        dim (int): A dimension along which Softmax will be computed (so every slice\n            along dim will sum to 1).\n\n    .. note::\n        This module doesn't work directly with NLLLoss,\n        which expects the Log to be computed between the Softmax and itself.\n        Use `LogSoftmax` instead (it's faster and has better numerical properties).\n\n    Examples::\n\n        >>> m = nn.Softmax(dim=1)\n        >>> input = torch.randn(2, 3)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "dim": {
                    "Type": "typing.Optional[int]",
                    "Default": "None"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "Tanh": {
            "Doc": "Applies the Hyperbolic Tangent (Tanh) function element-wise.\n\n    Tanh is defined as:\n\n    .. math::\n        \\text{Tanh}(x) = \\tanh(x) = \\frac{\\exp(x) - \\exp(-x)} {\\exp(x) + \\exp(-x)}\n\n    Shape:\n        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.\n        - Output: :math:`(*)`, same shape as the input.\n\n    .. image:: ../scripts/activation_images/Tanh.png\n\n    Examples::\n\n        >>> m = nn.Tanh()\n        >>> input = torch.randn(2)\n        >>> output = m(input)\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          }
        }
      },
      "batchnorm": {
        "Doc": null,
        "Classes": {
          "BatchNorm1d": {
            "Doc": "Applies Batch Normalization over a 2D or 3D input as described in the paper\n    `Batch Normalization: Accelerating Deep Network Training by Reducing\n    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .\n\n    .. math::\n\n        y = \\frac{x - \\mathrm{E}[x]}{\\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The mean and standard-deviation are calculated per-dimension over\n    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors\n    of size `C` (where `C` is the number of features or channels of the input). By default, the\n    elements of :math:`\\gamma` are set to 1 and the elements of :math:`\\beta` are set to 0. The\n    standard-deviation is calculated via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.\n\n    Also by default, during training this layer keeps running estimates of its\n    computed mean and variance, which are then used for normalization during\n    evaluation. The running estimates are kept with a default :attr:`momentum`\n    of 0.1.\n\n    If :attr:`track_running_stats` is set to ``False``, this layer then does not\n    keep running estimates, and batch statistics are instead used during\n    evaluation time as well.\n\n    .. note::\n        This :attr:`momentum` argument is different from one used in optimizer\n        classes and the conventional notion of momentum. Mathematically, the\n        update rule for running statistics here is\n        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,\n        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the\n        new observed value.\n\n    Because the Batch Normalization is done over the `C` dimension, computing statistics\n    on `(N, L)` slices, it's common terminology to call this Temporal Batch Normalization.\n\n    Args:\n        num_features: number of features or channels :math:`C` of the input\n        eps: a value added to the denominator for numerical stability.\n            Default: 1e-5\n        momentum: the value used for the running_mean and running_var\n            computation. Can be set to ``None`` for cumulative moving average\n            (i.e. simple average). Default: 0.1\n        affine: a boolean value that when set to ``True``, this module has\n            learnable affine parameters. Default: ``True``\n        track_running_stats: a boolean value that when set to ``True``, this\n            module tracks the running mean and variance, and when set to ``False``,\n            this module does not track such statistics, and initializes statistics\n            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.\n            When these buffers are ``None``, this module always uses batch statistics.\n            in both training and eval modes. Default: ``True``\n\n    Shape:\n        - Input: :math:`(N, C)` or :math:`(N, C, L)`, where :math:`N` is the batch size,\n          :math:`C` is the number of features or channels, and :math:`L` is the sequence length\n        - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)\n\n    Examples::\n\n        >>> # With Learnable Parameters\n        >>> m = nn.BatchNorm1d(100)\n        >>> # Without Learnable Parameters\n        >>> m = nn.BatchNorm1d(100, affine=False)\n        >>> input = torch.randn(20, 100)\n        >>> output = m(input)\n    "
          },
          "BatchNorm2d": {
            "Doc": "Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs\n    with additional channel dimension) as described in the paper\n    `Batch Normalization: Accelerating Deep Network Training by Reducing\n    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .\n\n    .. math::\n\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The mean and standard-deviation are calculated per-dimension over\n    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors\n    of size `C` (where `C` is the input size). By default, the elements of :math:`\\gamma` are set\n    to 1 and the elements of :math:`\\beta` are set to 0. The standard-deviation is calculated\n    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.\n\n    Also by default, during training this layer keeps running estimates of its\n    computed mean and variance, which are then used for normalization during\n    evaluation. The running estimates are kept with a default :attr:`momentum`\n    of 0.1.\n\n    If :attr:`track_running_stats` is set to ``False``, this layer then does not\n    keep running estimates, and batch statistics are instead used during\n    evaluation time as well.\n\n    .. note::\n        This :attr:`momentum` argument is different from one used in optimizer\n        classes and the conventional notion of momentum. Mathematically, the\n        update rule for running statistics here is\n        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,\n        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the\n        new observed value.\n\n    Because the Batch Normalization is done over the `C` dimension, computing statistics\n    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.\n\n    Args:\n        num_features: :math:`C` from an expected input of size\n            :math:`(N, C, H, W)`\n        eps: a value added to the denominator for numerical stability.\n            Default: 1e-5\n        momentum: the value used for the running_mean and running_var\n            computation. Can be set to ``None`` for cumulative moving average\n            (i.e. simple average). Default: 0.1\n        affine: a boolean value that when set to ``True``, this module has\n            learnable affine parameters. Default: ``True``\n        track_running_stats: a boolean value that when set to ``True``, this\n            module tracks the running mean and variance, and when set to ``False``,\n            this module does not track such statistics, and initializes statistics\n            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.\n            When these buffers are ``None``, this module always uses batch statistics.\n            in both training and eval modes. Default: ``True``\n\n    Shape:\n        - Input: :math:`(N, C, H, W)`\n        - Output: :math:`(N, C, H, W)` (same shape as input)\n\n    Examples::\n\n        >>> # With Learnable Parameters\n        >>> m = nn.BatchNorm2d(100)\n        >>> # Without Learnable Parameters\n        >>> m = nn.BatchNorm2d(100, affine=False)\n        >>> input = torch.randn(20, 100, 35, 45)\n        >>> output = m(input)\n    "
          },
          "BatchNorm3d": {
            "Doc": "Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs\n    with additional channel dimension) as described in the paper\n    `Batch Normalization: Accelerating Deep Network Training by Reducing\n    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .\n\n    .. math::\n\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The mean and standard-deviation are calculated per-dimension over\n    the mini-batches and :math:`\\gamma` and :math:`\\beta` are learnable parameter vectors\n    of size `C` (where `C` is the input size). By default, the elements of :math:`\\gamma` are set\n    to 1 and the elements of :math:`\\beta` are set to 0. The standard-deviation is calculated\n    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.\n\n    Also by default, during training this layer keeps running estimates of its\n    computed mean and variance, which are then used for normalization during\n    evaluation. The running estimates are kept with a default :attr:`momentum`\n    of 0.1.\n\n    If :attr:`track_running_stats` is set to ``False``, this layer then does not\n    keep running estimates, and batch statistics are instead used during\n    evaluation time as well.\n\n    .. note::\n        This :attr:`momentum` argument is different from one used in optimizer\n        classes and the conventional notion of momentum. Mathematically, the\n        update rule for running statistics here is\n        :math:`\\hat{x}_\\text{new} = (1 - \\text{momentum}) \\times \\hat{x} + \\text{momentum} \\times x_t`,\n        where :math:`\\hat{x}` is the estimated statistic and :math:`x_t` is the\n        new observed value.\n\n    Because the Batch Normalization is done over the `C` dimension, computing statistics\n    on `(N, D, H, W)` slices, it's common terminology to call this Volumetric Batch Normalization\n    or Spatio-temporal Batch Normalization.\n\n    Args:\n        num_features: :math:`C` from an expected input of size\n            :math:`(N, C, D, H, W)`\n        eps: a value added to the denominator for numerical stability.\n            Default: 1e-5\n        momentum: the value used for the running_mean and running_var\n            computation. Can be set to ``None`` for cumulative moving average\n            (i.e. simple average). Default: 0.1\n        affine: a boolean value that when set to ``True``, this module has\n            learnable affine parameters. Default: ``True``\n        track_running_stats: a boolean value that when set to ``True``, this\n            module tracks the running mean and variance, and when set to ``False``,\n            this module does not track such statistics, and initializes statistics\n            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.\n            When these buffers are ``None``, this module always uses batch statistics.\n            in both training and eval modes. Default: ``True``\n\n    Shape:\n        - Input: :math:`(N, C, D, H, W)`\n        - Output: :math:`(N, C, D, H, W)` (same shape as input)\n\n    Examples::\n\n        >>> # With Learnable Parameters\n        >>> m = nn.BatchNorm3d(100)\n        >>> # Without Learnable Parameters\n        >>> m = nn.BatchNorm3d(100, affine=False)\n        >>> input = torch.randn(20, 100, 35, 45, 10)\n        >>> output = m(input)\n    "
          }
        }
      },
      "conv": {
        "Doc": null,
        "Classes": {
          "Conv1d": {
            "Doc": "Applies a 1D convolution over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size\n    :math:`(N, C_{\\text{in}}, L)` and output :math:`(N, C_{\\text{out}}, L_{\\text{out}})` can be\n    precisely described as:\n\n    .. math::\n        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +\n        \\sum_{k = 0}^{C_{in} - 1} \\text{weight}(C_{\\text{out}_j}, k)\n        \\star \\text{input}(N_i, k)\n\n    where :math:`\\star` is the valid `cross-correlation`_ operator,\n    :math:`N` is a batch size, :math:`C` denotes a number of channels,\n    :math:`L` is a length of signal sequence.\n    \n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation, a single\n      number or a one-element tuple.\n\n    * :attr:`padding` controls the amount of padding applied to the input. It\n      can be either a string {'valid', 'same'} or a tuple of ints giving the\n      amount of implicit padding applied on both sides.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also\n      known as the \u00e0 trous algorithm. It is harder to describe, but this `link`_\n      has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    Note:\n        When `groups == in_channels` and `out_channels == K * in_channels`,\n        where `K` is a positive integer, this operation is also known as a \"depthwise convolution\".\n\n        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,\n        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments\n        :math:`(C_\\text{in}=C_\\text{in}, C_\\text{out}=C_\\text{in} \\times \\text{K}, ..., \\text{groups}=C_\\text{in})`.\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Note:\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n        the input so the output has the shape as the input. However, this mode\n        doesn't support any stride values other than 1.\n\n    Note:\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int, tuple or str, optional): Padding added to both sides of\n            the input. Default: 0\n        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,\n            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\n        dilation (int or tuple, optional): Spacing between kernel\n            elements. Default: 1\n        groups (int, optional): Number of blocked connections from input\n            channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the\n            output. Default: ``True``\n\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`\n        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where\n\n          .. math::\n              L_{out} = \\left\\lfloor\\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}\n                        \\times (\\text{kernel\\_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n            :math:`(\\text{out\\_channels},\n            \\frac{\\text{in\\_channels}}{\\text{groups}}, \\text{kernel\\_size})`.\n            The values of these weights are sampled from\n            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`\n        bias (Tensor):   the learnable bias of the module of shape\n            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are\n            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \\frac{groups}{C_\\text{in} * \\text{kernel\\_size}}`\n\n    Examples::\n\n        >>> m = nn.Conv1d(16, 33, 3, stride=2)\n        >>> input = torch.randn(20, 16, 50)\n        >>> output = m(input)\n\n    .. _cross-correlation:\n        https://en.wikipedia.org/wiki/Cross-correlation\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[str, int, typing.Tuple[int]]",
                    "Default": "0"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "1"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "Conv2d": {
            "Doc": "Applies a 2D convolution over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size\n    :math:`(N, C_{\\text{in}}, H, W)` and output :math:`(N, C_{\\text{out}}, H_{\\text{out}}, W_{\\text{out}})`\n    can be precisely described as:\n\n    .. math::\n        \\text{out}(N_i, C_{\\text{out}_j}) = \\text{bias}(C_{\\text{out}_j}) +\n        \\sum_{k = 0}^{C_{\\text{in}} - 1} \\text{weight}(C_{\\text{out}_j}, k) \\star \\text{input}(N_i, k)\n\n\n    where :math:`\\star` is the valid 2D `cross-correlation`_ operator,\n    :math:`N` is a batch size, :math:`C` denotes a number of channels,\n    :math:`H` is a height of input planes in pixels, and :math:`W` is\n    width in pixels.\n    \n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation, a single\n      number or a tuple.\n\n    * :attr:`padding` controls the amount of padding applied to the input. It\n      can be either a string {'valid', 'same'} or a tuple of ints giving the\n      amount of implicit padding applied on both sides.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also\n      known as the \u00e0 trous algorithm. It is harder to describe, but this `link`_\n      has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the height and width dimension\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n\n    Note:\n        When `groups == in_channels` and `out_channels == K * in_channels`,\n        where `K` is a positive integer, this operation is also known as a \"depthwise convolution\".\n\n        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,\n        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments\n        :math:`(C_\\text{in}=C_\\text{in}, C_\\text{out}=C_\\text{in} \\times \\text{K}, ..., \\text{groups}=C_\\text{in})`.\n\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Note:\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n        the input so the output has the shape as the input. However, this mode\n        doesn't support any stride values other than 1.\n\n    Note:\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int, tuple or str, optional): Padding added to all four sides of\n            the input. Default: 0\n        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,\n            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n        groups (int, optional): Number of blocked connections from input\n            channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the\n            output. Default: ``True``\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] - \\text{dilation}[0]\n                        \\times (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] - \\text{dilation}[1]\n                        \\times (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n            :math:`(\\text{out\\_channels}, \\frac{\\text{in\\_channels}}{\\text{groups}},`\n            :math:`\\text{kernel\\_size[0]}, \\text{kernel\\_size[1]})`.\n            The values of these weights are sampled from\n            :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{1}\\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape\n            (out_channels). If :attr:`bias` is ``True``,\n            then the values of these weights are\n            sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n            :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{1}\\text{kernel\\_size}[i]}`\n\n    Examples:\n\n        >>> # With square kernels and equal stride\n        >>> m = nn.Conv2d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\n        >>> # non-square kernels and unequal stride and with padding and dilation\n        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))\n        >>> input = torch.randn(20, 16, 50, 100)\n        >>> output = m(input)\n\n    .. _cross-correlation:\n        https://en.wikipedia.org/wiki/Cross-correlation\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[str, int, typing.Tuple[int, int]]",
                    "Default": "0"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "1"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "Conv3d": {
            "Doc": "Applies a 3D convolution over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C_{in}, D, H, W)`\n    and output :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` can be precisely described as:\n\n    .. math::\n        out(N_i, C_{out_j}) = bias(C_{out_j}) +\n                                \\sum_{k = 0}^{C_{in} - 1} weight(C_{out_j}, k) \\star input(N_i, k)\n\n    where :math:`\\star` is the valid 3D `cross-correlation`_ operator\n    \n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation.\n\n    * :attr:`padding` controls the amount of padding applied to the input. It\n      can be either a string {'valid', 'same'} or a tuple of ints giving the\n      amount of implicit padding applied on both sides.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.\n      It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\n          the second `int` for the height dimension and the third `int` for the width dimension\n\n    Note:\n        When `groups == in_channels` and `out_channels == K * in_channels`,\n        where `K` is a positive integer, this operation is also known as a \"depthwise convolution\".\n\n        In other words, for an input of size :math:`(N, C_{in}, L_{in})`,\n        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments\n        :math:`(C_\\text{in}=C_\\text{in}, C_\\text{out}=C_\\text{in} \\times \\text{K}, ..., \\text{groups}=C_\\text{in})`.\n\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Note:\n        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads\n        the input so the output has the shape as the input. However, this mode\n        doesn't support any stride values other than 1.\n\n    Note:\n        This module supports complex data types i.e. ``complex32, complex64, complex128``.\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int, tuple or str, optional): Padding added to all six sides of\n            the input. Default: 0\n        padding_mode (str, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or :math:`(C_{out}, D_{out}, H_{out}, W_{out})`,\n          where\n\n          .. math::\n              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0]\n                    \\times (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1]\n                    \\times (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{dilation}[2]\n                    \\times (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n                         :math:`(\\text{out\\_channels}, \\frac{\\text{in\\_channels}}{\\text{groups}},`\n                         :math:`\\text{kernel\\_size[0]}, \\text{kernel\\_size[1]}, \\text{kernel\\_size[2]})`.\n                         The values of these weights are sampled from\n                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{2}\\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,\n                         then the values of these weights are\n                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{in} * \\prod_{i=0}^{2}\\text{kernel\\_size}[i]}`\n\n    Examples::\n\n        >>> # With square kernels and equal stride\n        >>> m = nn.Conv3d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))\n        >>> input = torch.randn(20, 16, 10, 50, 100)\n        >>> output = m(input)\n\n    .. _cross-correlation:\n        https://en.wikipedia.org/wiki/Cross-correlation\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[str, int, typing.Tuple[int, int, int]]",
                    "Default": "0"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "1"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "ConvTranspose1d": {
            "Doc": "Applies a 1D transposed convolution operator over an input image\n    composed of several input planes.\n\n    This module can be seen as the gradient of Conv1d with respect to its input.\n    It is also known as a fractionally-strided convolution or\n    a deconvolution (although it is not an actual deconvolution operation as it does\n    not compute a true inverse of convolution). For more information, see the visualizations\n    `here`_ and the `Deconvolutional Networks`_ paper.\n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation.\n\n    * :attr:`padding` controls the amount of implicit zero padding on both\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\n      below for details.\n\n    * :attr:`output_padding` controls the additional size added to one side\n      of the output shape. See note below for details.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    Note:\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\n        amount of zero padding to both sizes of the input. This is set so that\n        when a :class:`~torch.nn.Conv1d` and a :class:`~torch.nn.ConvTranspose1d`\n        are initialized with same parameters, they are inverses of each other in\n        regard to the input and output shapes. However, when ``stride > 1``,\n        :class:`~torch.nn.Conv1d` maps multiple input shapes to the same output\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\n        effectively increasing the calculated output shape on one side. Note\n        that :attr:`output_padding` is only used to find output shape, but does\n        not actually add zero-padding to output.\n\n    Note:\n        In some circumstances when using the CUDA backend with CuDNN, this operator\n        may select a nondeterministic algorithm to increase performance. If this is\n        undesirable, you can try to make the operation deterministic (potentially at\n        a performance cost) by setting ``torch.backends.cudnn.deterministic =\n        True``.\n        Please see the notes on :doc:`/notes/randomness` for background.\n\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding\n            will be added to both sides of the input. Default: 0\n        output_padding (int or tuple, optional): Additional size added to one side\n            of the output shape. Default: 0\n        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, L_{in})` or :math:`(C_{in}, L_{in})`\n        - Output: :math:`(N, C_{out}, L_{out})` or :math:`(C_{out}, L_{out})`, where\n\n          .. math::\n              L_{out} = (L_{in} - 1) \\times \\text{stride} - 2 \\times \\text{padding} + \\text{dilation}\n                        \\times (\\text{kernel\\_size} - 1) + \\text{output\\_padding} + 1\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n                         :math:`(\\text{in\\_channels}, \\frac{\\text{out\\_channels}}{\\text{groups}},`\n                         :math:`\\text{kernel\\_size})`.\n                         The values of these weights are sampled from\n                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\text{kernel\\_size}}`\n        bias (Tensor):   the learnable bias of the module of shape (out_channels).\n                         If :attr:`bias` is ``True``, then the values of these weights are\n                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\text{kernel\\_size}}`\n\n    .. _`here`:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n\n    .. _`Deconvolutional Networks`:\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "0"
                  },
                  "output_padding": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "0"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "1"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "output_size": {
                    "Type": "typing.Optional[typing.List[int]]",
                    "Default": "None"
                  }
                }
              }
            }
          },
          "ConvTranspose2d": {
            "Doc": "Applies a 2D transposed convolution operator over an input image\n    composed of several input planes.\n\n    This module can be seen as the gradient of Conv2d with respect to its input.\n    It is also known as a fractionally-strided convolution or\n    a deconvolution (although it is not an actual deconvolution operation as it does\n    not compute a true inverse of convolution). For more information, see the visualizations\n    `here`_ and the `Deconvolutional Networks`_ paper.\n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation.\n\n    * :attr:`padding` controls the amount of implicit zero padding on both\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\n      below for details.\n\n    * :attr:`output_padding` controls the additional size added to one side\n      of the output shape. See note below for details.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`\n    can either be:\n\n        - a single ``int`` -- in which case the same value is used for the height and width dimensions\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n\n    Note:\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\n        amount of zero padding to both sizes of the input. This is set so that\n        when a :class:`~torch.nn.Conv2d` and a :class:`~torch.nn.ConvTranspose2d`\n        are initialized with same parameters, they are inverses of each other in\n        regard to the input and output shapes. However, when ``stride > 1``,\n        :class:`~torch.nn.Conv2d` maps multiple input shapes to the same output\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\n        effectively increasing the calculated output shape on one side. Note\n        that :attr:`output_padding` is only used to find output shape, but does\n        not actually add zero-padding to output.\n\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding\n            will be added to both sides of each dimension in the input. Default: 0\n        output_padding (int or tuple, optional): Additional size added to one side\n            of each dimension in the output shape. Default: 0\n        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, H_{in}, W_{in})` or :math:`(C_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` or :math:`(C_{out}, H_{out}, W_{out})`, where\n\n        .. math::\n              H_{out} = (H_{in} - 1) \\times \\text{stride}[0] - 2 \\times \\text{padding}[0] + \\text{dilation}[0]\n                        \\times (\\text{kernel\\_size}[0] - 1) + \\text{output\\_padding}[0] + 1\n        .. math::\n              W_{out} = (W_{in} - 1) \\times \\text{stride}[1] - 2 \\times \\text{padding}[1] + \\text{dilation}[1]\n                        \\times (\\text{kernel\\_size}[1] - 1) + \\text{output\\_padding}[1] + 1\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n                         :math:`(\\text{in\\_channels}, \\frac{\\text{out\\_channels}}{\\text{groups}},`\n                         :math:`\\text{kernel\\_size[0]}, \\text{kernel\\_size[1]})`.\n                         The values of these weights are sampled from\n                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{1}\\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape (out_channels)\n                         If :attr:`bias` is ``True``, then the values of these weights are\n                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{1}\\text{kernel\\_size}[i]}`\n\n    Examples::\n\n        >>> # With square kernels and equal stride\n        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))\n        >>> input = torch.randn(20, 16, 50, 100)\n        >>> output = m(input)\n        >>> # exact output size can be also specified as an argument\n        >>> input = torch.randn(1, 16, 12, 12)\n        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)\n        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)\n        >>> h = downsample(input)\n        >>> h.size()\n        torch.Size([1, 16, 6, 6])\n        >>> output = upsample(h, output_size=input.size())\n        >>> output.size()\n        torch.Size([1, 16, 12, 12])\n\n    .. _`here`:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n\n    .. _`Deconvolutional Networks`:\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "0"
                  },
                  "output_padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "0"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "1"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "output_size": {
                    "Type": "typing.Optional[typing.List[int]]",
                    "Default": "None"
                  }
                }
              }
            }
          },
          "ConvTranspose3d": {
            "Doc": "Applies a 3D transposed convolution operator over an input image composed of several input\n    planes.\n    The transposed convolution operator multiplies each input value element-wise by a learnable kernel,\n    and sums over the outputs from all input feature planes.\n\n    This module can be seen as the gradient of Conv3d with respect to its input.\n    It is also known as a fractionally-strided convolution or\n    a deconvolution (although it is not an actual deconvolution operation as it does\n    not compute a true inverse of convolution). For more information, see the visualizations\n    `here`_ and the `Deconvolutional Networks`_ paper.\n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    * :attr:`stride` controls the stride for the cross-correlation.\n\n    * :attr:`padding` controls the amount of implicit zero padding on both\n      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note\n      below for details.\n\n    * :attr:`output_padding` controls the additional size added to one side\n      of the output shape. See note below for details.\n\n    * :attr:`dilation` controls the spacing between the kernel points; also known as the \u00e0 trous algorithm.\n      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.\n\n    * :attr:`groups` controls the connections between inputs and outputs.\n      :attr:`in_channels` and :attr:`out_channels` must both be divisible by\n      :attr:`groups`. For example,\n\n        * At groups=1, all inputs are convolved to all outputs.\n        * At groups=2, the operation becomes equivalent to having two conv\n          layers side by side, each seeing half the input channels\n          and producing half the output channels, and both subsequently\n          concatenated.\n        * At groups= :attr:`in_channels`, each input channel is convolved with\n          its own set of filters (of size\n          :math:`\\frac{\\text{out\\_channels}}{\\text{in\\_channels}}`).\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`\n    can either be:\n\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\n          the second `int` for the height dimension and the third `int` for the width dimension\n\n    Note:\n        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``\n        amount of zero padding to both sizes of the input. This is set so that\n        when a :class:`~torch.nn.Conv3d` and a :class:`~torch.nn.ConvTranspose3d`\n        are initialized with same parameters, they are inverses of each other in\n        regard to the input and output shapes. However, when ``stride > 1``,\n        :class:`~torch.nn.Conv3d` maps multiple input shapes to the same output\n        shape. :attr:`output_padding` is provided to resolve this ambiguity by\n        effectively increasing the calculated output shape on one side. Note\n        that :attr:`output_padding` is only used to find output shape, but does\n        not actually add zero-padding to output.\n\n    Note:\n        In some circumstances when given tensors on a CUDA device and using CuDNN, this operator may select a nondeterministic algorithm to increase performance. If this is undesirable, you can try to make the operation deterministic (potentially at a performance cost) by setting ``torch.backends.cudnn.deterministic = True``. See :doc:`/notes/randomness` for more information.\n\n    Args:\n        in_channels (int): Number of channels in the input image\n        out_channels (int): Number of channels produced by the convolution\n        kernel_size (int or tuple): Size of the convolving kernel\n        stride (int or tuple, optional): Stride of the convolution. Default: 1\n        padding (int or tuple, optional): ``dilation * (kernel_size - 1) - padding`` zero-padding\n            will be added to both sides of each dimension in the input. Default: 0\n        output_padding (int or tuple, optional): Additional size added to one side\n            of each dimension in the output shape. Default: 0\n        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1\n        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``\n        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1\n    \n\n    Shape:\n        - Input: :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` or :math:`(C_{in}, D_{in}, H_{in}, W_{in})`\n        - Output: :math:`(N, C_{out}, D_{out}, H_{out}, W_{out})` or\n          :math:`(C_{out}, D_{out}, H_{out}, W_{out})`, where\n\n        .. math::\n              D_{out} = (D_{in} - 1) \\times \\text{stride}[0] - 2 \\times \\text{padding}[0] + \\text{dilation}[0]\n                        \\times (\\text{kernel\\_size}[0] - 1) + \\text{output\\_padding}[0] + 1\n        .. math::\n              H_{out} = (H_{in} - 1) \\times \\text{stride}[1] - 2 \\times \\text{padding}[1] + \\text{dilation}[1]\n                        \\times (\\text{kernel\\_size}[1] - 1) + \\text{output\\_padding}[1] + 1\n        .. math::\n              W_{out} = (W_{in} - 1) \\times \\text{stride}[2] - 2 \\times \\text{padding}[2] + \\text{dilation}[2]\n                        \\times (\\text{kernel\\_size}[2] - 1) + \\text{output\\_padding}[2] + 1\n\n\n    Attributes:\n        weight (Tensor): the learnable weights of the module of shape\n                         :math:`(\\text{in\\_channels}, \\frac{\\text{out\\_channels}}{\\text{groups}},`\n                         :math:`\\text{kernel\\_size[0]}, \\text{kernel\\_size[1]}, \\text{kernel\\_size[2]})`.\n                         The values of these weights are sampled from\n                         :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{2}\\text{kernel\\_size}[i]}`\n        bias (Tensor):   the learnable bias of the module of shape (out_channels)\n                         If :attr:`bias` is ``True``, then the values of these weights are\n                         sampled from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                         :math:`k = \\frac{groups}{C_\\text{out} * \\prod_{i=0}^{2}\\text{kernel\\_size}[i]}`\n\n    Examples::\n\n        >>> # With square kernels and equal stride\n        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)\n        >>> # non-square kernels and unequal stride and with padding\n        >>> m = nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))\n        >>> input = torch.randn(20, 16, 10, 50, 100)\n        >>> output = m(input)\n\n    .. _`here`:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n\n    .. _`Deconvolutional Networks`:\n        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "1"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "0"
                  },
                  "output_padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "0"
                  },
                  "groups": {
                    "Type": "<class 'int'>",
                    "Default": "1"
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "dilation": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "1"
                  },
                  "padding_mode": {
                    "Type": "<class 'str'>",
                    "Default": "zeros"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "output_size": {
                    "Type": "typing.Optional[typing.List[int]]",
                    "Default": "None"
                  }
                }
              }
            }
          }
        }
      },
      "dropout": {
        "Doc": null,
        "Classes": {
          "Dropout": {
            "Doc": "During training, randomly zeroes some of the elements of the input\n    tensor with probability :attr:`p` using samples from a Bernoulli\n    distribution. Each channel will be zeroed out independently on every forward\n    call.\n\n    This has proven to be an effective technique for regularization and\n    preventing the co-adaptation of neurons as described in the paper\n    `Improving neural networks by preventing co-adaptation of feature\n    detectors`_ .\n\n    Furthermore, the outputs are scaled by a factor of :math:`\\frac{1}{1-p}` during\n    training. This means that during evaluation the module simply computes an\n    identity function.\n\n    Args:\n        p: probability of an element to be zeroed. Default: 0.5\n        inplace: If set to ``True``, will do this operation in-place. Default: ``False``\n\n    Shape:\n        - Input: :math:`(*)`. Input can be of any shape\n        - Output: :math:`(*)`. Output is of the same shape as input\n\n    Examples::\n\n        >>> m = nn.Dropout(p=0.2)\n        >>> input = torch.randn(20, 16)\n        >>> output = m(input)\n\n    .. _Improving neural networks by preventing co-adaptation of feature\n        detectors: https://arxiv.org/abs/1207.0580\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          }
        }
      },
      "linear": {
        "Doc": null,
        "Classes": {
          "Bilinear": {
            "Doc": "Applies a bilinear transformation to the incoming data:\n    :math:`y = x_1^T A x_2 + b`\n\n    Args:\n        in1_features: size of each first input sample\n        in2_features: size of each second input sample\n        out_features: size of each output sample\n        bias: If set to False, the layer will not learn an additive bias.\n            Default: ``True``\n\n    Shape:\n        - Input1: :math:`(*, H_{in1})` where :math:`H_{in1}=\\text{in1\\_features}` and\n          :math:`*` means any number of additional dimensions including none. All but the last dimension\n          of the inputs should be the same.\n        - Input2: :math:`(*, H_{in2})` where :math:`H_{in2}=\\text{in2\\_features}`.\n        - Output: :math:`(*, H_{out})` where :math:`H_{out}=\\text{out\\_features}`\n          and all but the last dimension are the same shape as the input.\n\n    Attributes:\n        weight: the learnable weights of the module of shape\n            :math:`(\\text{out\\_features}, \\text{in1\\_features}, \\text{in2\\_features})`.\n            The values are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where\n            :math:`k = \\frac{1}{\\text{in1\\_features}}`\n        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.\n                If :attr:`bias` is ``True``, the values are initialized from\n                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where\n                :math:`k = \\frac{1}{\\text{in1\\_features}}`\n\n    Examples::\n\n        >>> m = nn.Bilinear(20, 30, 40)\n        >>> input1 = torch.randn(128, 20)\n        >>> input2 = torch.randn(128, 30)\n        >>> output = m(input1, input2)\n        >>> print(output.size())\n        torch.Size([128, 40])\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in1_features": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "in2_features": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_features": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input1": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "input2": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              },
              "reset_parameters": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              }
            }
          },
          "Linear": {
            "Doc": "Applies a linear transformation to the incoming data: :math:`y = xA^T + b`\n\n    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.\n\n    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.\n\n    Args:\n        in_features: size of each input sample\n        out_features: size of each output sample\n        bias: If set to ``False``, the layer will not learn an additive bias.\n            Default: ``True``\n\n    Shape:\n        - Input: :math:`(*, H_{in})` where :math:`*` means any number of\n          dimensions including none and :math:`H_{in} = \\text{in\\_features}`.\n        - Output: :math:`(*, H_{out})` where all but the last dimension\n          are the same shape as the input and :math:`H_{out} = \\text{out\\_features}`.\n\n    Attributes:\n        weight: the learnable weights of the module of shape\n            :math:`(\\text{out\\_features}, \\text{in\\_features})`. The values are\n            initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`, where\n            :math:`k = \\frac{1}{\\text{in\\_features}}`\n        bias:   the learnable bias of the module of shape :math:`(\\text{out\\_features})`.\n                If :attr:`bias` is ``True``, the values are initialized from\n                :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` where\n                :math:`k = \\frac{1}{\\text{in\\_features}}`\n\n    Examples::\n\n        >>> m = nn.Linear(20, 30)\n        >>> input = torch.randn(128, 20)\n        >>> output = m(input)\n        >>> print(output.size())\n        torch.Size([128, 30])\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "in_features": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "out_features": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "bias": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              },
              "reset_parameters": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              }
            }
          }
        }
      },
      "normalization": {
        "Doc": null,
        "Classes": {
          "GroupNorm": {
            "Doc": "Applies Group Normalization over a mini-batch of inputs as described in\n    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__\n\n    .. math::\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The input channels are separated into :attr:`num_groups` groups, each containing\n    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by\n    :attr:`num_groups`. The mean and standard-deviation are calculated\n    separately over the each group. :math:`\\gamma` and :math:`\\beta` are learnable\n    per-channel affine transform parameter vectors of size :attr:`num_channels` if\n    :attr:`affine` is ``True``.\n    The standard-deviation is calculated via the biased estimator, equivalent to\n    `torch.var(input, unbiased=False)`.\n\n    This layer uses statistics computed from input data in both training and\n    evaluation modes.\n\n    Args:\n        num_groups (int): number of groups to separate the channels into\n        num_channels (int): number of channels expected in input\n        eps: a value added to the denominator for numerical stability. Default: 1e-5\n        affine: a boolean value that when set to ``True``, this module\n            has learnable per-channel affine parameters initialized to ones (for weights)\n            and zeros (for biases). Default: ``True``.\n\n    Shape:\n        - Input: :math:`(N, C, *)` where :math:`C=\\text{num\\_channels}`\n        - Output: :math:`(N, C, *)` (same shape as input)\n\n    Examples::\n\n        >>> input = torch.randn(20, 6, 10, 10)\n        >>> # Separate 6 channels into 3 groups\n        >>> m = nn.GroupNorm(3, 6)\n        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)\n        >>> m = nn.GroupNorm(6, 6)\n        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)\n        >>> m = nn.GroupNorm(1, 6)\n        >>> # Activating the module\n        >>> output = m(input)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "num_groups": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "num_channels": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "eps": {
                    "Type": "<class 'float'>",
                    "Default": "1e-05"
                  },
                  "affine": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              },
              "reset_parameters": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              }
            }
          },
          "LayerNorm": {
            "Doc": "Applies Layer Normalization over a mini-batch of inputs as described in\n    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__\n\n    .. math::\n        y = \\frac{x - \\mathrm{E}[x]}{ \\sqrt{\\mathrm{Var}[x] + \\epsilon}} * \\gamma + \\beta\n\n    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`\n    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`\n    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over\n    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).\n    :math:`\\gamma` and :math:`\\beta` are learnable affine transform parameters of\n    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.\n    The standard-deviation is calculated via the biased estimator, equivalent to\n    `torch.var(input, unbiased=False)`.\n\n    .. note::\n        Unlike Batch Normalization and Instance Normalization, which applies\n        scalar scale and bias for each entire channel/plane with the\n        :attr:`affine` option, Layer Normalization applies per-element scale and\n        bias with :attr:`elementwise_affine`.\n\n    This layer uses statistics computed from input data in both training and\n    evaluation modes.\n\n    Args:\n        normalized_shape (int or list or torch.Size): input shape from an expected input\n            of size\n\n            .. math::\n                [* \\times \\text{normalized\\_shape}[0] \\times \\text{normalized\\_shape}[1]\n                    \\times \\ldots \\times \\text{normalized\\_shape}[-1]]\n\n            If a single integer is used, it is treated as a singleton list, and this module will\n            normalize over the last dimension which is expected to be of that specific size.\n        eps: a value added to the denominator for numerical stability. Default: 1e-5\n        elementwise_affine: a boolean value that when set to ``True``, this module\n            has learnable per-element affine parameters initialized to ones (for weights)\n            and zeros (for biases). Default: ``True``.\n\n    Attributes:\n        weight: the learnable weights of the module of shape\n            :math:`\\text{normalized\\_shape}` when :attr:`elementwise_affine` is set to ``True``.\n            The values are initialized to 1.\n        bias:   the learnable bias of the module of shape\n                :math:`\\text{normalized\\_shape}` when :attr:`elementwise_affine` is set to ``True``.\n                The values are initialized to 0.\n\n    Shape:\n        - Input: :math:`(N, *)`\n        - Output: :math:`(N, *)` (same shape as input)\n\n    Examples::\n\n        >>> # NLP Example\n        >>> batch, sentence_length, embedding_dim = 20, 5, 10\n        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)\n        >>> layer_norm = nn.LayerNorm(embedding_dim)\n        >>> # Activate module\n        >>> layer_norm(embedding)\n        >>>\n        >>> # Image Example\n        >>> N, C, H, W = 20, 5, 10, 10\n        >>> input = torch.randn(N, C, H, W)\n        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)\n        >>> # as shown in the image below\n        >>> layer_norm = nn.LayerNorm([C, H, W])\n        >>> output = layer_norm(input)\n\n    .. image:: ../_static/img/nn/layer_norm.jpg\n        :scale: 50 %\n\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "normalized_shape": {
                    "Type": "typing.Union[int, typing.List[int], torch.Size]",
                    "Default": null
                  },
                  "eps": {
                    "Type": "<class 'float'>",
                    "Default": "1e-05"
                  },
                  "elementwise_affine": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "extra_repr": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              },
              "reset_parameters": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  }
                }
              }
            }
          }
        }
      },
      "pooling": {
        "Doc": null,
        "Classes": {
          "AdaptiveAvgPool1d": {
            "Doc": "Applies a 1D adaptive average pooling over an input signal composed of several input planes.\n\n    The output size is :math:`L_{out}`, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size :math:`L_{out}`.\n\n    Shape:\n        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.\n        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where\n          :math:`L_{out}=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5\n        >>> m = nn.AdaptiveAvgPool1d(5)\n        >>> input = torch.randn(1, 64, 8)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AdaptiveAvgPool2d": {
            "Doc": "Applies a 2D adaptive average pooling over an input signal composed of several input planes.\n\n    The output is of size H x W, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size of the image of the form H x W.\n                     Can be a tuple (H, W) or a single H for a square image H x H.\n                     H and W can be either a ``int``, or ``None`` which means the size will\n                     be the same as that of the input.\n\n    Shape:\n        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, S_{0}, S_{1})` or :math:`(C, S_{0}, S_{1})`, where\n          :math:`S=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5x7\n        >>> m = nn.AdaptiveAvgPool2d((5,7))\n        >>> input = torch.randn(1, 64, 8, 9)\n        >>> output = m(input)\n        >>> # target output size of 7x7 (square)\n        >>> m = nn.AdaptiveAvgPool2d(7)\n        >>> input = torch.randn(1, 64, 10, 9)\n        >>> output = m(input)\n        >>> # target output size of 10x7\n        >>> m = nn.AdaptiveAvgPool2d((None, 7))\n        >>> input = torch.randn(1, 64, 10, 9)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AdaptiveAvgPool3d": {
            "Doc": "Applies a 3D adaptive average pooling over an input signal composed of several input planes.\n\n    The output is of size D x H x W, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size of the form D x H x W.\n                     Can be a tuple (D, H, W) or a single number D for a cube D x D x D.\n                     D, H and W can be either a ``int``, or ``None`` which means the size will\n                     be the same as that of the input.\n\n    Shape:\n        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, S_{0}, S_{1}, S_{2})` or :math:`(C, S_{0}, S_{1}, S_{2})`,\n          where :math:`S=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5x7x9\n        >>> m = nn.AdaptiveAvgPool3d((5,7,9))\n        >>> input = torch.randn(1, 64, 8, 9, 10)\n        >>> output = m(input)\n        >>> # target output size of 7x7x7 (cube)\n        >>> m = nn.AdaptiveAvgPool3d(7)\n        >>> input = torch.randn(1, 64, 10, 9, 8)\n        >>> output = m(input)\n        >>> # target output size of 7x9x8\n        >>> m = nn.AdaptiveAvgPool3d((7, None, None))\n        >>> input = torch.randn(1, 64, 10, 9, 8)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AdaptiveMaxPool1d": {
            "Doc": "Applies a 1D adaptive max pooling over an input signal composed of several input planes.\n\n    The output size is :math:`L_{out}`, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size :math:`L_{out}`.\n        return_indices: if ``True``, will return the indices along with the outputs.\n                        Useful to pass to nn.MaxUnpool1d. Default: ``False``\n\n    Shape:\n        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.\n        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where\n          :math:`L_{out}=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5\n        >>> m = nn.AdaptiveMaxPool1d(5)\n        >>> input = torch.randn(1, 64, 8)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AdaptiveMaxPool2d": {
            "Doc": "Applies a 2D adaptive max pooling over an input signal composed of several input planes.\n\n    The output is of size :math:`H_{out} \\times W_{out}`, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size of the image of the form :math:`H_{out} \\times W_{out}`.\n                     Can be a tuple :math:`(H_{out}, W_{out})` or a single :math:`H_{out}` for a\n                     square image :math:`H_{out} \\times H_{out}`. :math:`H_{out}` and :math:`W_{out}`\n                     can be either a ``int``, or ``None`` which means the size will be the same as that\n                     of the input.\n        return_indices: if ``True``, will return the indices along with the outputs.\n                        Useful to pass to nn.MaxUnpool2d. Default: ``False``\n\n    Shape:\n        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where\n          :math:`(H_{out}, W_{out})=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5x7\n        >>> m = nn.AdaptiveMaxPool2d((5,7))\n        >>> input = torch.randn(1, 64, 8, 9)\n        >>> output = m(input)\n        >>> # target output size of 7x7 (square)\n        >>> m = nn.AdaptiveMaxPool2d(7)\n        >>> input = torch.randn(1, 64, 10, 9)\n        >>> output = m(input)\n        >>> # target output size of 10x7\n        >>> m = nn.AdaptiveMaxPool2d((None, 7))\n        >>> input = torch.randn(1, 64, 10, 9)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AdaptiveMaxPool3d": {
            "Doc": "Applies a 3D adaptive max pooling over an input signal composed of several input planes.\n\n    The output is of size :math:`D_{out} \\times H_{out} \\times W_{out}`, for any input size.\n    The number of output features is equal to the number of input planes.\n\n    Args:\n        output_size: the target output size of the image of the form :math:`D_{out} \\times H_{out} \\times W_{out}`.\n                     Can be a tuple :math:`(D_{out}, H_{out}, W_{out})` or a single\n                     :math:`D_{out}` for a cube :math:`D_{out} \\times D_{out} \\times D_{out}`.\n                     :math:`D_{out}`, :math:`H_{out}` and :math:`W_{out}` can be either a\n                     ``int``, or ``None`` which means the size will be the same as that of the input.\n\n        return_indices: if ``True``, will return the indices along with the outputs.\n                        Useful to pass to nn.MaxUnpool3d. Default: ``False``\n\n    Shape:\n        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`,\n          where :math:`(D_{out}, H_{out}, W_{out})=\\text{output\\_size}`.\n\n    Examples:\n        >>> # target output size of 5x7x9\n        >>> m = nn.AdaptiveMaxPool3d((5,7,9))\n        >>> input = torch.randn(1, 64, 8, 9, 10)\n        >>> output = m(input)\n        >>> # target output size of 7x7x7 (cube)\n        >>> m = nn.AdaptiveMaxPool3d(7)\n        >>> input = torch.randn(1, 64, 10, 9, 8)\n        >>> output = m(input)\n        >>> # target output size of 7x9x8\n        >>> m = nn.AdaptiveMaxPool3d((7, None, None))\n        >>> input = torch.randn(1, 64, 10, 9, 8)\n        >>> output = m(input)\n\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AvgPool1d": {
            "Doc": "Applies a 1D average pooling over an input signal composed of several\n    input planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`,\n    output :math:`(N, C, L_{out})` and :attr:`kernel_size` :math:`k`\n    can be precisely described as:\n\n    .. math::\n\n        \\text{out}(N_i, C_j, l) = \\frac{1}{k} \\sum_{m=0}^{k-1}\n                               \\text{input}(N_i, C_j, \\text{stride} \\times l + m)\n\n    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides\n    for :attr:`padding` number of points.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can each be\n    an ``int`` or a one-element tuple.\n\n    Args:\n        kernel_size: the size of the window\n        stride: the stride of the window. Default value is :attr:`kernel_size`\n        padding: implicit zero padding to be added on both sides\n        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape\n        count_include_pad: when True, will include the zero-padding in the averaging calculation\n\n    Shape:\n        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.\n        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where\n\n          .. math::\n              L_{out} = \\left\\lfloor \\frac{L_{in} +\n              2 \\times \\text{padding} - \\text{kernel\\_size}}{\\text{stride}} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool with window of size=3, stride=2\n        >>> m = nn.AvgPool1d(3, stride=2)\n        >>> m(torch.tensor([[[1.,2,3,4,5,6,7]]]))\n        tensor([[[2., 4., 6.]]])\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "None"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int]]",
                    "Default": "0"
                  },
                  "ceil_mode": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "count_include_pad": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AvgPool2d": {
            "Doc": "Applies a 2D average pooling over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,\n    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`\n    can be precisely described as:\n\n    .. math::\n\n        out(N_i, C_j, h, w)  = \\frac{1}{kH * kW} \\sum_{m=0}^{kH-1} \\sum_{n=0}^{kW-1}\n                               input(N_i, C_j, stride[0] \\times h + m, stride[1] \\times w + n)\n\n    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides\n    for :attr:`padding` number of points.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the height and width dimension\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n\n    Args:\n        kernel_size: the size of the window\n        stride: the stride of the window. Default value is :attr:`kernel_size`\n        padding: implicit zero padding to be added on both sides\n        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape\n        count_include_pad: when True, will include the zero-padding in the averaging calculation\n        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.\n\n\n    Shape:\n        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in}  + 2 \\times \\text{padding}[0] -\n                \\text{kernel\\_size}[0]}{\\text{stride}[0]} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in}  + 2 \\times \\text{padding}[1] -\n                \\text{kernel\\_size}[1]}{\\text{stride}[1]} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool of square window of size=3, stride=2\n        >>> m = nn.AvgPool2d(3, stride=2)\n        >>> # pool of non-square window\n        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))\n        >>> input = torch.randn(20, 16, 50, 32)\n        >>> output = m(input)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int], NoneType]",
                    "Default": "None"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int]]",
                    "Default": "0"
                  },
                  "ceil_mode": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "count_include_pad": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "divisor_override": {
                    "Type": "typing.Optional[int]",
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "AvgPool3d": {
            "Doc": "Applies a 3D average pooling over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,\n    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`\n    can be precisely described as:\n\n    .. math::\n        \\begin{aligned}\n            \\text{out}(N_i, C_j, d, h, w) ={} & \\sum_{k=0}^{kD-1} \\sum_{m=0}^{kH-1} \\sum_{n=0}^{kW-1} \\\\\n                                              & \\frac{\\text{input}(N_i, C_j, \\text{stride}[0] \\times d + k,\n                                                      \\text{stride}[1] \\times h + m, \\text{stride}[2] \\times w + n)}\n                                                     {kD \\times kH \\times kW}\n        \\end{aligned}\n\n    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on all three sides\n    for :attr:`padding` number of points.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    The parameters :attr:`kernel_size`, :attr:`stride` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\n          the second `int` for the height dimension and the third `int` for the width dimension\n\n    Args:\n        kernel_size: the size of the window\n        stride: the stride of the window. Default value is :attr:`kernel_size`\n        padding: implicit zero padding to be added on all three sides\n        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape\n        count_include_pad: when True, will include the zero-padding in the averaging calculation\n        divisor_override: if specified, it will be used as divisor, otherwise :attr:`kernel_size` will be used\n\n    Shape:\n        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or\n          :math:`(C, D_{out}, H_{out}, W_{out})`, where\n\n          .. math::\n              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] -\n                    \\text{kernel\\_size}[0]}{\\text{stride}[0]} + 1\\right\\rfloor\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] -\n                    \\text{kernel\\_size}[1]}{\\text{stride}[1]} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] -\n                    \\text{kernel\\_size}[2]}{\\text{stride}[2]} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool of square window of size=3, stride=2\n        >>> m = nn.AvgPool3d(3, stride=2)\n        >>> # pool of non-square window\n        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))\n        >>> input = torch.randn(20, 16, 50,44, 31)\n        >>> output = m(input)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "kernel_size": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": null
                  },
                  "stride": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int], NoneType]",
                    "Default": "None"
                  },
                  "padding": {
                    "Type": "typing.Union[int, typing.Tuple[int, int, int]]",
                    "Default": "0"
                  },
                  "ceil_mode": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "count_include_pad": {
                    "Type": "<class 'bool'>",
                    "Default": "True"
                  },
                  "divisor_override": {
                    "Type": "typing.Optional[int]",
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "MaxPool1d": {
            "Doc": "Applies a 1D max pooling over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, L)`\n    and output :math:`(N, C, L_{out})` can be precisely described as:\n\n    .. math::\n        out(N_i, C_j, k) = \\max_{m=0, \\ldots, \\text{kernel\\_size} - 1}\n                input(N_i, C_j, stride \\times k + m)\n\n    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides\n    for :attr:`padding` number of points. :attr:`dilation` is the stride between the elements within the\n    sliding window. This `link`_ has a nice visualization of the pooling parameters.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    Args:\n        kernel_size: The size of the sliding window, must be > 0.\n        stride: The stride of the sliding window, must be > 0. Default value is :attr:`kernel_size`.\n        padding: Implicit negative infinity padding to be added on both sides, must be >= 0 and <= kernel_size / 2.\n        dilation: The stride between elements within a sliding window, must be > 0.\n        return_indices: If ``True``, will return the argmax along with the max values.\n                        Useful for :class:`torch.nn.MaxUnpool1d` later\n        ceil_mode: If ``True``, will use `ceil` instead of `floor` to compute the output shape. This\n                   ensures that every element in the input tensor is covered by a sliding window.\n\n    Shape:\n        - Input: :math:`(N, C, L_{in})` or :math:`(C, L_{in})`.\n        - Output: :math:`(N, C, L_{out})` or :math:`(C, L_{out})`, where\n\n          .. math::\n              L_{out} = \\left\\lfloor \\frac{L_{in} + 2 \\times \\text{padding} - \\text{dilation}\n                    \\times (\\text{kernel\\_size} - 1) - 1}{\\text{stride}} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool of size=3, stride=2\n        >>> m = nn.MaxPool1d(3, stride=2)\n        >>> input = torch.randn(20, 16, 50)\n        >>> output = m(input)\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "MaxPool2d": {
            "Doc": "Applies a 2D max pooling over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,\n    output :math:`(N, C, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kH, kW)`\n    can be precisely described as:\n\n    .. math::\n        \\begin{aligned}\n            out(N_i, C_j, h, w) ={} & \\max_{m=0, \\ldots, kH-1} \\max_{n=0, \\ldots, kW-1} \\\\\n                                    & \\text{input}(N_i, C_j, \\text{stride[0]} \\times h + m,\n                                                   \\text{stride[1]} \\times w + n)\n        \\end{aligned}\n\n    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides\n    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.\n    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the height and width dimension\n        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,\n          and the second `int` for the width dimension\n\n    Args:\n        kernel_size: the size of the window to take a max over\n        stride: the stride of the window. Default value is :attr:`kernel_size`\n        padding: implicit zero padding to be added on both sides\n        dilation: a parameter that controls the stride of elements in the window\n        return_indices: if ``True``, will return the max indices along with the outputs.\n                        Useful for :class:`torch.nn.MaxUnpool2d` later\n        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape\n\n    Shape:\n        - Input: :math:`(N, C, H_{in}, W_{in})` or :math:`(C, H_{in}, W_{in})`\n        - Output: :math:`(N, C, H_{out}, W_{out})` or :math:`(C, H_{out}, W_{out})`, where\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 * \\text{padding[0]} - \\text{dilation[0]}\n                    \\times (\\text{kernel\\_size[0]} - 1) - 1}{\\text{stride[0]}} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 * \\text{padding[1]} - \\text{dilation[1]}\n                    \\times (\\text{kernel\\_size[1]} - 1) - 1}{\\text{stride[1]}} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool of square window of size=3, stride=2\n        >>> m = nn.MaxPool2d(3, stride=2)\n        >>> # pool of non-square window\n        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))\n        >>> input = torch.randn(20, 16, 50, 32)\n        >>> output = m(input)\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          },
          "MaxPool3d": {
            "Doc": "Applies a 3D max pooling over an input signal composed of several input\n    planes.\n\n    In the simplest case, the output value of the layer with input size :math:`(N, C, D, H, W)`,\n    output :math:`(N, C, D_{out}, H_{out}, W_{out})` and :attr:`kernel_size` :math:`(kD, kH, kW)`\n    can be precisely described as:\n\n    .. math::\n        \\begin{aligned}\n            \\text{out}(N_i, C_j, d, h, w) ={} & \\max_{k=0, \\ldots, kD-1} \\max_{m=0, \\ldots, kH-1} \\max_{n=0, \\ldots, kW-1} \\\\\n                                              & \\text{input}(N_i, C_j, \\text{stride[0]} \\times d + k,\n                                                             \\text{stride[1]} \\times h + m, \\text{stride[2]} \\times w + n)\n        \\end{aligned}\n\n    If :attr:`padding` is non-zero, then the input is implicitly padded with negative infinity on both sides\n    for :attr:`padding` number of points. :attr:`dilation` controls the spacing between the kernel points.\n    It is harder to describe, but this `link`_ has a nice visualization of what :attr:`dilation` does.\n\n    Note:\n        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding\n        or the input. Sliding windows that would start in the right padded region are ignored.\n\n    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:\n\n        - a single ``int`` -- in which case the same value is used for the depth, height and width dimension\n        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,\n          the second `int` for the height dimension and the third `int` for the width dimension\n\n    Args:\n        kernel_size: the size of the window to take a max over\n        stride: the stride of the window. Default value is :attr:`kernel_size`\n        padding: implicit zero padding to be added on all three sides\n        dilation: a parameter that controls the stride of elements in the window\n        return_indices: if ``True``, will return the max indices along with the outputs.\n                        Useful for :class:`torch.nn.MaxUnpool3d` later\n        ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape\n\n    Shape:\n        - Input: :math:`(N, C, D_{in}, H_{in}, W_{in})` or :math:`(C, D_{in}, H_{in}, W_{in})`.\n        - Output: :math:`(N, C, D_{out}, H_{out}, W_{out})` or :math:`(C, D_{out}, H_{out}, W_{out})`, where\n\n          .. math::\n              D_{out} = \\left\\lfloor\\frac{D_{in} + 2 \\times \\text{padding}[0] - \\text{dilation}[0] \\times\n                (\\text{kernel\\_size}[0] - 1) - 1}{\\text{stride}[0]} + 1\\right\\rfloor\n\n          .. math::\n              H_{out} = \\left\\lfloor\\frac{H_{in} + 2 \\times \\text{padding}[1] - \\text{dilation}[1] \\times\n                (\\text{kernel\\_size}[1] - 1) - 1}{\\text{stride}[1]} + 1\\right\\rfloor\n\n          .. math::\n              W_{out} = \\left\\lfloor\\frac{W_{in} + 2 \\times \\text{padding}[2] - \\text{dilation}[2] \\times\n                (\\text{kernel\\_size}[2] - 1) - 1}{\\text{stride}[2]} + 1\\right\\rfloor\n\n    Examples::\n\n        >>> # pool of square window of size=3, stride=2\n        >>> m = nn.MaxPool3d(3, stride=2)\n        >>> # pool of non-square window\n        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))\n        >>> input = torch.randn(20, 16, 50,44, 31)\n        >>> output = m(input)\n\n    .. _link:\n        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md\n    ",
            "Functions": {
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  }
                }
              }
            }
          }
        }
      },
      "rnn": {
        "Doc": null,
        "Classes": {
          "GRU": {
            "Doc": "Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.\n\n\n    For each element in the input sequence, each layer computes the following\n    function:\n\n    .. math::\n        \\begin{array}{ll}\n            r_t = \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\\n            z_t = \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\\n            n_t = \\tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\\\\n            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}\n        \\end{array}\n\n    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input\n    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer\n    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,\n    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.\n    :math:`\\sigma` is the sigmoid function, and :math:`*` is the Hadamard product.\n\n    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer\n    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by\n    dropout :math:`\\delta^{(l-1)}_t` where each :math:`\\delta^{(l-1)}_t` is a Bernoulli random\n    variable which is :math:`0` with probability :attr:`dropout`.\n\n    Args:\n        input_size: The number of expected features in the input `x`\n        hidden_size: The number of features in the hidden state `h`\n        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``\n            would mean stacking two GRUs together to form a `stacked GRU`,\n            with the second GRU taking in outputs of the first GRU and\n            computing the final results. Default: 1\n        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.\n            Default: ``True``\n        batch_first: If ``True``, then the input and output tensors are provided\n            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.\n            Note that this does not apply to hidden or cell states. See the\n            Inputs/Outputs sections below for details.  Default: ``False``\n        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each\n            GRU layer except the last layer, with dropout probability equal to\n            :attr:`dropout`. Default: 0\n        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``\n\n    Inputs: input, h_0\n        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,\n          :math:`(L, N, H_{in})` when ``batch_first=False`` or\n          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of\n          the input sequence.  The input can also be a packed variable length sequence.\n          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or\n          :func:`torch.nn.utils.rnn.pack_sequence` for details.\n        * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})`\n          containing the initial hidden state for the input sequence. Defaults to zeros if not provided.\n\n        where:\n\n        .. math::\n            \\begin{aligned}\n                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{out} ={} & \\text{hidden\\_size}\n            \\end{aligned}\n\n    Outputs: output, h_n\n        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,\n          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or\n          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features\n          `(h_t)` from the last layer of the GRU, for each `t`. If a\n          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output\n          will also be a packed sequence.\n        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the final hidden state\n          for the input sequence.\n\n    Attributes:\n        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\\text{k}^{th}` layer\n            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.\n            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`\n        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\\text{k}^{th}` layer\n            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`\n        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\\text{k}^{th}` layer\n            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`\n        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\\text{k}^{th}` layer\n            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`\n\n    .. note::\n        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`\n        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`\n\n    .. note::\n        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.\n        Example of splitting the output layers when ``batch_first=False``:\n        ``output.view(seq_len, batch, num_directions, hidden_size)``.\n\n    .. note::\n        ``batch_first`` argument is ignored for unbatched inputs.\n\n    .. include:: ../cudnn_persistent_rnn.rst\n\n    Examples::\n\n        >>> rnn = nn.GRU(10, 20, 2)\n        >>> input = torch.randn(5, 3, 10)\n        >>> h0 = torch.randn(2, 3, 20)\n        >>> output, hn = rnn(input, h0)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "*args": {
                    "Type": null,
                    "Default": null
                  },
                  "**kwargs": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": null,
                    "Default": null
                  },
                  "hx": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              }
            }
          },
          "LSTM": {
            "Doc": "Applies a multi-layer long short-term memory (LSTM) RNN to an input\n    sequence.\n\n\n    For each element in the input sequence, each layer computes the following\n    function:\n\n    .. math::\n        \\begin{array}{ll} \\\\\n            i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\\n            f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\\n            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\\n            o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\\n            c_t = f_t \\odot c_{t-1} + i_t \\odot g_t \\\\\n            h_t = o_t \\odot \\tanh(c_t) \\\\\n        \\end{array}\n\n    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell\n    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`\n    is the hidden state of the layer at time `t-1` or the initial hidden\n    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,\n    :math:`o_t` are the input, forget, cell, and output gates, respectively.\n    :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.\n\n    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer\n    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by\n    dropout :math:`\\delta^{(l-1)}_t` where each :math:`\\delta^{(l-1)}_t` is a Bernoulli random\n    variable which is :math:`0` with probability :attr:`dropout`.\n\n    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes\n    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from\n    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).\n    Second, the output hidden state of each layer will be multiplied by a learnable projection\n    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output\n    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact\n    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.\n\n    Args:\n        input_size: The number of expected features in the input `x`\n        hidden_size: The number of features in the hidden state `h`\n        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``\n            would mean stacking two LSTMs together to form a `stacked LSTM`,\n            with the second LSTM taking in outputs of the first LSTM and\n            computing the final results. Default: 1\n        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.\n            Default: ``True``\n        batch_first: If ``True``, then the input and output tensors are provided\n            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.\n            Note that this does not apply to hidden or cell states. See the\n            Inputs/Outputs sections below for details.  Default: ``False``\n        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each\n            LSTM layer except the last layer, with dropout probability equal to\n            :attr:`dropout`. Default: 0\n        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``\n        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0\n\n    Inputs: input, (h_0, c_0)\n        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,\n          :math:`(L, N, H_{in})` when ``batch_first=False`` or\n          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of\n          the input sequence.  The input can also be a packed variable length sequence.\n          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or\n          :func:`torch.nn.utils.rnn.pack_sequence` for details.\n        * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the\n          initial hidden state for each element in the input sequence.\n          Defaults to zeros if (h_0, c_0) is not provided.\n        * **c_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{cell})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the\n          initial cell state for each element in the input sequence.\n          Defaults to zeros if (h_0, c_0) is not provided.\n\n        where:\n\n        .. math::\n            \\begin{aligned}\n                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{cell} ={} & \\text{hidden\\_size} \\\\\n                H_{out} ={} & \\text{proj\\_size if } \\text{proj\\_size}>0 \\text{ otherwise hidden\\_size} \\\\\n            \\end{aligned}\n\n    Outputs: output, (h_n, c_n)\n        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,\n          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or\n          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features\n          `(h_t)` from the last layer of the LSTM, for each `t`. If a\n          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output\n          will also be a packed sequence. When ``bidirectional=True``, `output` will contain\n          a concatenation of the forward and reverse hidden states at each time step in the sequence.\n        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the\n          final hidden state for each element in the sequence. When ``bidirectional=True``,\n          `h_n` will contain a concatenation of the final forward and reverse hidden states, respectively.\n        * **c_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{cell})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the\n          final cell state for each element in the sequence. When ``bidirectional=True``,\n          `c_n` will contain a concatenation of the final forward and reverse cell states, respectively.\n\n    Attributes:\n        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\\text{k}^{th}` layer\n            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.\n            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If\n            ``proj_size > 0`` was specified, the shape will be\n            `(4*hidden_size, num_directions * proj_size)` for `k > 0`\n        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\\text{k}^{th}` layer\n            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``\n            was specified, the shape will be `(4*hidden_size, proj_size)`.\n        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\\text{k}^{th}` layer\n            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`\n        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\\text{k}^{th}` layer\n            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`\n        weight_hr_l[k] : the learnable projection weights of the :math:`\\text{k}^{th}` layer\n            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was\n            specified.\n        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.\n            Only present when ``bidirectional=True``.\n        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.\n            Only present when ``bidirectional=True``.\n        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.\n            Only present when ``bidirectional=True``.\n        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.\n            Only present when ``bidirectional=True``.\n        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.\n            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.\n\n    .. note::\n        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`\n        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`\n\n    .. note::\n        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.\n        Example of splitting the output layers when ``batch_first=False``:\n        ``output.view(seq_len, batch, num_directions, hidden_size)``.\n\n    .. note::\n        For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`; the\n        former contains the final forward and reverse hidden states, while the latter contains the\n        final forward hidden state and the initial reverse hidden state.\n\n    .. note::\n        ``batch_first`` argument is ignored for unbatched inputs.\n\n    .. include:: ../cudnn_rnn_determinism.rst\n\n    .. include:: ../cudnn_persistent_rnn.rst\n\n    Examples::\n\n        >>> rnn = nn.LSTM(10, 20, 2)\n        >>> input = torch.randn(5, 3, 10)\n        >>> h0 = torch.randn(2, 3, 20)\n        >>> c0 = torch.randn(2, 3, 20)\n        >>> output, (hn, cn) = rnn(input, (h0, c0))\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "*args": {
                    "Type": null,
                    "Default": null
                  },
                  "**kwargs": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "check_forward_args": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "hidden": {
                    "Type": "typing.Tuple[torch.Tensor, torch.Tensor]",
                    "Default": null
                  },
                  "batch_sizes": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": null,
                    "Default": null
                  },
                  "hx": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "get_expected_cell_size": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "batch_sizes": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": null
                  }
                }
              },
              "permute_hidden": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "hx": {
                    "Type": "typing.Tuple[torch.Tensor, torch.Tensor]",
                    "Default": null
                  },
                  "permutation": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": null
                  }
                }
              }
            }
          },
          "RNN": {
            "Doc": "Applies a multi-layer Elman RNN with :math:`\\tanh` or :math:`\\text{ReLU}` non-linearity to an\n    input sequence.\n\n\n    For each element in the input sequence, each layer computes the following\n    function:\n\n    .. math::\n        h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})\n\n    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is\n    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the\n    previous layer at time `t-1` or the initial hidden state at time `0`.\n    If :attr:`nonlinearity` is ``'relu'``, then :math:`\\text{ReLU}` is used instead of :math:`\\tanh`.\n\n    Args:\n        input_size: The number of expected features in the input `x`\n        hidden_size: The number of features in the hidden state `h`\n        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``\n            would mean stacking two RNNs together to form a `stacked RNN`,\n            with the second RNN taking in outputs of the first RNN and\n            computing the final results. Default: 1\n        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``\n        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.\n            Default: ``True``\n        batch_first: If ``True``, then the input and output tensors are provided\n            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.\n            Note that this does not apply to hidden or cell states. See the\n            Inputs/Outputs sections below for details.  Default: ``False``\n        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each\n            RNN layer except the last layer, with dropout probability equal to\n            :attr:`dropout`. Default: 0\n        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``\n\n    Inputs: input, h_0\n        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,\n          :math:`(L, N, H_{in})` when ``batch_first=False`` or\n          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of\n          the input sequence.  The input can also be a packed variable length sequence.\n          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or\n          :func:`torch.nn.utils.rnn.pack_sequence` for details.\n        * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the initial hidden\n          state for the input sequence batch. Defaults to zeros if not provided.\n\n        where:\n\n        .. math::\n            \\begin{aligned}\n                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{out} ={} & \\text{hidden\\_size}\n            \\end{aligned}\n\n    Outputs: output, h_n\n        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,\n          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or\n          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features\n          `(h_t)` from the last layer of the RNN, for each `t`. If a\n          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output\n          will also be a packed sequence.\n        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or\n          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the final hidden state\n          for each element in the batch.\n\n    Attributes:\n        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,\n            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is\n            `(hidden_size, num_directions * hidden_size)`\n        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,\n            of shape `(hidden_size, hidden_size)`\n        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,\n            of shape `(hidden_size)`\n        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,\n            of shape `(hidden_size)`\n\n    .. note::\n        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`\n        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`\n\n    .. note::\n        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.\n        Example of splitting the output layers when ``batch_first=False``:\n        ``output.view(seq_len, batch, num_directions, hidden_size)``.\n\n    .. note::\n        ``batch_first`` argument is ignored for unbatched inputs.\n\n    .. include:: ../cudnn_rnn_determinism.rst\n\n    .. include:: ../cudnn_persistent_rnn.rst\n\n    Examples::\n\n        >>> rnn = nn.RNN(10, 20, 2)\n        >>> input = torch.randn(5, 3, 10)\n        >>> h0 = torch.randn(2, 3, 20)\n        >>> output, hn = rnn(input, h0)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "*args": {
                    "Type": null,
                    "Default": null
                  },
                  "**kwargs": {
                    "Type": null,
                    "Default": null
                  }
                }
              },
              "forward": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "input": {
                    "Type": null,
                    "Default": null
                  },
                  "hx": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              }
            }
          }
        },
        "Functions": {
          "apply_permutation": {
            "Doc": null,
            "Args": {
              "tensor": {
                "Type": "<class 'torch.Tensor'>",
                "Default": null
              },
              "permutation": {
                "Type": "<class 'torch.Tensor'>",
                "Default": null
              },
              "dim": {
                "Type": "<class 'int'>",
                "Default": "1"
              }
            }
          },
          "overload": {
            "Doc": "Decorator for overloaded functions/methods.\n\n    In a stub file, place two or more stub definitions for the same\n    function in a row, each decorated with @overload.  For example:\n\n      @overload\n      def utf8(value: None) -> None: ...\n      @overload\n      def utf8(value: bytes) -> bytes: ...\n      @overload\n      def utf8(value: str) -> bytes: ...\n\n    In a non-stub file (i.e. a regular .py file), do the same but\n    follow it with an implementation.  The implementation should *not*\n    be decorated with @overload.  For example:\n\n      @overload\n      def utf8(value: None) -> None: ...\n      @overload\n      def utf8(value: bytes) -> bytes: ...\n      @overload\n      def utf8(value: str) -> bytes: ...\n      def utf8(value):\n          # implementation goes here\n    ",
            "Args": {
              "func": {
                "Type": null,
                "Default": null
              }
            }
          }
        }
      },
      "transformer": {
        "Doc": null,
        "Classes": {
          "Transformer": {
            "Doc": "A transformer model. User is able to modify the attributes as needed. The architecture\n    is based on the paper \"Attention Is All You Need\". Ashish Vaswani, Noam Shazeer,\n    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and\n    Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information\n    Processing Systems, pages 6000-6010.\n\n    Args:\n        d_model: the number of expected features in the encoder/decoder inputs (default=512).\n        nhead: the number of heads in the multiheadattention models (default=8).\n        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).\n        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).\n        dim_feedforward: the dimension of the feedforward network model (default=2048).\n        dropout: the dropout value (default=0.1).\n        activation: the activation function of encoder/decoder intermediate layer, can be a string\n            (\"relu\" or \"gelu\") or a unary callable. Default: relu\n        custom_encoder: custom encoder (default=None).\n        custom_decoder: custom decoder (default=None).\n        layer_norm_eps: the eps value in layer normalization components (default=1e-5).\n        batch_first: If ``True``, then the input and output tensors are provided\n            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).\n        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before\n            other attention and feedforward operations, otherwise after. Default: ``False`` (after).\n\n    Examples::\n        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)\n        >>> src = torch.rand((10, 32, 512))\n        >>> tgt = torch.rand((20, 32, 512))\n        >>> out = transformer_model(src, tgt)\n\n    Note: A full example to apply nn.Transformer module for the word language model is available in\n    https://github.com/pytorch/examples/tree/master/word_language_model\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "d_model": {
                    "Type": "<class 'int'>",
                    "Default": "512"
                  },
                  "nhead": {
                    "Type": "<class 'int'>",
                    "Default": "8"
                  },
                  "num_encoder_layers": {
                    "Type": "<class 'int'>",
                    "Default": "6"
                  },
                  "num_decoder_layers": {
                    "Type": "<class 'int'>",
                    "Default": "6"
                  },
                  "dim_feedforward": {
                    "Type": "<class 'int'>",
                    "Default": "2048"
                  },
                  "dropout": {
                    "Type": "<class 'float'>",
                    "Default": "0.1"
                  },
                  "activation": {
                    "Type": "typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]",
                    "Default": "<function relu at 0x0000025958C71DC0>"
                  },
                  "custom_encoder": {
                    "Type": "typing.Optional[typing.Any]",
                    "Default": "None"
                  },
                  "custom_decoder": {
                    "Type": "typing.Optional[typing.Any]",
                    "Default": "None"
                  },
                  "layer_norm_eps": {
                    "Type": "<class 'float'>",
                    "Default": "1e-05"
                  },
                  "batch_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "norm_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": "Take in and process masked source/target sequences.\n\n        Args:\n            src: the sequence to the encoder (required).\n            tgt: the sequence to the decoder (required).\n            src_mask: the additive mask for the src sequence (optional).\n            tgt_mask: the additive mask for the tgt sequence (optional).\n            memory_mask: the additive mask for the encoder output (optional).\n            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).\n            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).\n            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).\n\n        Shape:\n            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or\n              `(N, S, E)` if `batch_first=True`.\n            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or\n              `(N, T, E)` if `batch_first=True`.\n            - src_mask: :math:`(S, S)` or :math:`(N\\cdot\\text{num\\_heads}, S, S)`.\n            - tgt_mask: :math:`(T, T)` or :math:`(N\\cdot\\text{num\\_heads}, T, T)`.\n            - memory_mask: :math:`(T, S)`.\n            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.\n            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.\n            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.\n\n            Note: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked\n            positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend\n            while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``\n            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor\n            is provided, it will be added to the attention weight.\n            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by\n            the attention. If a ByteTensor is provided, the non-zero positions will be ignored while the zero\n            positions will be unchanged. If a BoolTensor is provided, the positions with the\n            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.\n\n            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or\n              `(N, T, E)` if `batch_first=True`.\n\n            Note: Due to the multi-head attention architecture in the transformer model,\n            the output sequence length of a transformer is same as the input sequence\n            (i.e. target) length of the decoder.\n\n            where S is the source sequence length, T is the target sequence length, N is the\n            batch size, E is the feature number\n\n        Examples:\n            >>> # xdoctest: +SKIP\n            >>> output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)\n        ",
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "src": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "tgt": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "src_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "tgt_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "src_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "tgt_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  }
                }
              },
              "generate_square_subsequent_mask": {
                "Doc": "Generate a square mask for the sequence. The masked positions are filled with float('-inf').\n            Unmasked positions are filled with float(0.0).\n        ",
                "Args": {
                  "sz": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "device": {
                    "Type": null,
                    "Default": "cpu"
                  }
                }
              }
            }
          },
          "TransformerDecoder": {
            "Doc": "TransformerDecoder is a stack of N decoder layers\n\n    Args:\n        decoder_layer: an instance of the TransformerDecoderLayer() class (required).\n        num_layers: the number of sub-decoder-layers in the decoder (required).\n        norm: the layer normalization component (optional).\n\n    Examples::\n        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)\n        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)\n        >>> memory = torch.rand(10, 32, 512)\n        >>> tgt = torch.rand(20, 32, 512)\n        >>> out = transformer_decoder(tgt, memory)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "decoder_layer": {
                    "Type": null,
                    "Default": null
                  },
                  "num_layers": {
                    "Type": null,
                    "Default": null
                  },
                  "norm": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": "Pass the inputs (and mask) through the decoder layer in turn.\n\n        Args:\n            tgt: the sequence to the decoder (required).\n            memory: the sequence from the last layer of the encoder (required).\n            tgt_mask: the mask for the tgt sequence (optional).\n            memory_mask: the mask for the memory sequence (optional).\n            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).\n            memory_key_padding_mask: the mask for the memory keys per batch (optional).\n\n        Shape:\n            see the docs in Transformer class.\n        ",
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "tgt": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "memory": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "tgt_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "tgt_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  }
                }
              }
            }
          },
          "TransformerDecoderLayer": {
            "Doc": "TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.\n    This standard decoder layer is based on the paper \"Attention Is All You Need\".\n    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,\n    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in\n    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement\n    in a different way during application.\n\n    Args:\n        d_model: the number of expected features in the input (required).\n        nhead: the number of heads in the multiheadattention models (required).\n        dim_feedforward: the dimension of the feedforward network model (default=2048).\n        dropout: the dropout value (default=0.1).\n        activation: the activation function of the intermediate layer, can be a string\n            (\"relu\" or \"gelu\") or a unary callable. Default: relu\n        layer_norm_eps: the eps value in layer normalization components (default=1e-5).\n        batch_first: If ``True``, then the input and output tensors are provided\n            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).\n        norm_first: if ``True``, layer norm is done prior to self attention, multihead\n            attention and feedforward operations, respectively. Otherwise it's done after.\n            Default: ``False`` (after).\n\n    Examples::\n        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)\n        >>> memory = torch.rand(10, 32, 512)\n        >>> tgt = torch.rand(20, 32, 512)\n        >>> out = decoder_layer(tgt, memory)\n\n    Alternatively, when ``batch_first`` is ``True``:\n        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)\n        >>> memory = torch.rand(32, 10, 512)\n        >>> tgt = torch.rand(32, 20, 512)\n        >>> out = decoder_layer(tgt, memory)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "d_model": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "nhead": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "dim_feedforward": {
                    "Type": "<class 'int'>",
                    "Default": "2048"
                  },
                  "dropout": {
                    "Type": "<class 'float'>",
                    "Default": "0.1"
                  },
                  "activation": {
                    "Type": "typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]",
                    "Default": "<function relu at 0x0000025958C71DC0>"
                  },
                  "layer_norm_eps": {
                    "Type": "<class 'float'>",
                    "Default": "1e-05"
                  },
                  "batch_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "norm_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": "Pass the inputs (and mask) through the decoder layer.\n\n        Args:\n            tgt: the sequence to the decoder layer (required).\n            memory: the sequence from the last layer of the encoder (required).\n            tgt_mask: the mask for the tgt sequence (optional).\n            memory_mask: the mask for the memory sequence (optional).\n            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).\n            memory_key_padding_mask: the mask for the memory keys per batch (optional).\n\n        Shape:\n            see the docs in Transformer class.\n        ",
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "tgt": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "memory": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "tgt_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "tgt_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "memory_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  }
                }
              }
            }
          },
          "TransformerEncoder": {
            "Doc": "TransformerEncoder is a stack of N encoder layers. Users can build the\n    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.\n\n    Args:\n        encoder_layer: an instance of the TransformerEncoderLayer() class (required).\n        num_layers: the number of sub-encoder-layers in the encoder (required).\n        norm: the layer normalization component (optional).\n        enable_nested_tensor: if True, input will automatically convert to nested tensor\n            (and convert back on output). This will improve the overall performance of\n            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).\n\n    Examples::\n        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)\n        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)\n        >>> src = torch.rand(10, 32, 512)\n        >>> out = transformer_encoder(src)\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "encoder_layer": {
                    "Type": null,
                    "Default": null
                  },
                  "num_layers": {
                    "Type": null,
                    "Default": null
                  },
                  "norm": {
                    "Type": null,
                    "Default": "None"
                  },
                  "enable_nested_tensor": {
                    "Type": null,
                    "Default": "True"
                  },
                  "mask_check": {
                    "Type": null,
                    "Default": "True"
                  }
                }
              },
              "forward": {
                "Doc": "Pass the input through the encoder layers in turn.\n\n        Args:\n            src: the sequence to the encoder (required).\n            mask: the mask for the src sequence (optional).\n            src_key_padding_mask: the mask for the src keys per batch (optional).\n\n        Shape:\n            see the docs in Transformer class.\n        ",
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "src": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "src_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  }
                }
              }
            }
          },
          "TransformerEncoderLayer": {
            "Doc": "TransformerEncoderLayer is made up of self-attn and feedforward network.\n    This standard encoder layer is based on the paper \"Attention Is All You Need\".\n    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,\n    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in\n    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement\n    in a different way during application.\n\n    Args:\n        d_model: the number of expected features in the input (required).\n        nhead: the number of heads in the multiheadattention models (required).\n        dim_feedforward: the dimension of the feedforward network model (default=2048).\n        dropout: the dropout value (default=0.1).\n        activation: the activation function of the intermediate layer, can be a string\n            (\"relu\" or \"gelu\") or a unary callable. Default: relu\n        layer_norm_eps: the eps value in layer normalization components (default=1e-5).\n        batch_first: If ``True``, then the input and output tensors are provided\n            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).\n        norm_first: if ``True``, layer norm is done prior to attention and feedforward\n            operations, respectively. Otherwise it's done after. Default: ``False`` (after).\n\n    Examples::\n        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)\n        >>> src = torch.rand(10, 32, 512)\n        >>> out = encoder_layer(src)\n\n    Alternatively, when ``batch_first`` is ``True``:\n        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)\n        >>> src = torch.rand(32, 10, 512)\n        >>> out = encoder_layer(src)\n\n    Fast path:\n        forward() will use a special optimized implementation if all of the following\n        conditions are met:\n\n        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor\n          argument ``requires_grad``\n        - training is disabled (using ``.eval()``)\n        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)\n        - activation is one of: ``\"relu\"``, ``\"gelu\"``, ``torch.functional.relu``, or ``torch.functional.gelu``\n        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed\n        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``\n          nor ``src_key_padding_mask`` is passed\n        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case\n          unless the caller has manually modified one without modifying the other)\n\n        If the optimized implementation is in use, a\n        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be\n        passed for ``src`` to represent padding more efficiently than using a padding\n        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be\n        returned, and an additional speedup proportional to the fraction of the input that\n        is padding can be expected.\n    ",
            "Functions": {
              "__init__": {
                "Doc": null,
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "d_model": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "nhead": {
                    "Type": "<class 'int'>",
                    "Default": null
                  },
                  "dim_feedforward": {
                    "Type": "<class 'int'>",
                    "Default": "2048"
                  },
                  "dropout": {
                    "Type": "<class 'float'>",
                    "Default": "0.1"
                  },
                  "activation": {
                    "Type": "typing.Union[str, typing.Callable[[torch.Tensor], torch.Tensor]]",
                    "Default": "<function relu at 0x0000025958C71DC0>"
                  },
                  "layer_norm_eps": {
                    "Type": "<class 'float'>",
                    "Default": "1e-05"
                  },
                  "batch_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "norm_first": {
                    "Type": "<class 'bool'>",
                    "Default": "False"
                  },
                  "device": {
                    "Type": null,
                    "Default": "None"
                  },
                  "dtype": {
                    "Type": null,
                    "Default": "None"
                  }
                }
              },
              "forward": {
                "Doc": "Pass the input through the encoder layer.\n\n        Args:\n            src: the sequence to the encoder layer (required).\n            src_mask: the mask for the src sequence (optional).\n            src_key_padding_mask: the mask for the src keys per batch (optional).\n\n        Shape:\n            see the docs in Transformer class.\n        ",
                "Args": {
                  "self": {
                    "Type": null,
                    "Default": null
                  },
                  "src": {
                    "Type": "<class 'torch.Tensor'>",
                    "Default": null
                  },
                  "src_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  },
                  "src_key_padding_mask": {
                    "Type": "typing.Optional[torch.Tensor]",
                    "Default": "None"
                  }
                }
              }
            }
          }
        }
      }
    },
    "custom": {}
}

const urls = {
    auth: 'api/login',
    reg: 'api/registration',
    build: 'api/create_code',
    add_project: "/api/add_project",
}

export default {urls, nodes}