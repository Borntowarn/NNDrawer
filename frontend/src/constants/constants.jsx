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
    'Custom' : {}
}

const urls = {
    auth: 'api/login',
    reg: 'api/registration',
}

export default {urls, nodes}