"""
Transducer loss implementation (depends on numba)

Authors
 * Abdelwahab Heba 2020
"""

import paddle
from paddle.autograd import PyLayer
from paddle.nn import Layer

try:
    from numba import cuda
except ImportError:
    err_msg = "The optional dependency Numba is needed to use this module\n"
    err_msg += "Cannot import numba. To use Transducer loss\n"
    err_msg += "Please follow the instructions below\n"
    err_msg += "=============================\n"
    err_msg += "If you use your localhost:\n"
    err_msg += "pip install numba\n"
    err_msg += "export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/' \n"
    err_msg += "export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so' \n"
    err_msg += "================================ \n"
    err_msg += "If you use conda:\n"
    err_msg += "conda install numba cudatoolkit=9.0"
    raise ImportError(err_msg)

import math


@cuda.jit(
    "(float32[:,:,:,:], int32[:,:], float32[:,:,:], float32[:], int32[:], int32[:], int32, int32[:,:])"
)
def cu_kernel_forward(log_probs, labels, alpha, log_p, T, U, blank, lock):
    """
    Compute forward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    alpha : tensor
        3D Tensor of (batch x TimeLength x LabelLength) for forward computation.
    log_p : tensor
        1D Tensor of (batch) for forward cost computation.
    T : tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank indice.
    lock : tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """

    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = 0
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the previous computation of Alpha[b,U-1,:]
        # Do the computation over the whole Time sequence on alpha[B,U,:]
        # and then unlock the target U+1 for computation
        while t < T[b]:
            if u == 0:
                if t > 0:
                    alpha[b, t, 0] = (
                        alpha[b, t - 1, 0] + log_probs[b, t - 1, 0, blank]
                    )
                cuda.atomic.add(lock, (b, u + 1), -1)
                t += 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == 0:
                        alpha[b, 0, u] = (
                            alpha[b, 0, u - 1]
                            + log_probs[b, 0, u - 1, labels[b, u - 1]]
                        )
                    else:
                        # compute emission prob
                        emit = (
                            alpha[b, t, u - 1]
                            + log_probs[b, t, u - 1, labels[b, u - 1]]
                        )
                        # compute no_emission prob
                        no_emit = (
                            alpha[b, t - 1, u] + log_probs[b, t - 1, u, blank]
                        )
                        # do logsumexp between log_emit and log_no_emit
                        alpha[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u < U[b]:
                        cuda.atomic.add(lock, (b, u + 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)
                    t += 1
        if u == U[b]:
            # for each thread b (utterance)
            # normalize the loss over time
            log_p[b] = (
                alpha[b, T[b] - 1, U[b]] + log_probs[b, T[b] - 1, U[b], blank]
            ) / T[b]


@cuda.jit(
    "(float32[:,:,:,:], int32[:,:], float32[:,:,:], float32[:], int32[:], int32[:], int32, int32[:,:])"
)
def cu_kernel_backward(log_probs, labels, beta, log_p, T, U, blank, lock):
    """
    Compute backward pass for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    beta : tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    log_p : tensor
        1D Tensor of (batch) for backward cost computation.
    T : tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank indice.
    lock : tensor
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """
    # parallelize the forward algorithm over batch and target length dim
    b = cuda.blockIdx.x
    u = cuda.threadIdx.x
    t = T[b] - 1
    if u <= U[b]:
        # for each (B,U) Thread
        # wait the unlock of the next computation of beta[b,U+1,:]
        # Do the computation over the whole Time sequence on beta[B,U,:]
        # and then unlock the target U-1 for computation
        while t >= 0:
            if u == U[b]:
                if t == T[b] - 1:
                    beta[b, t, u] = log_probs[b, t, u, blank]
                else:
                    beta[b, t, u] = (
                        beta[b, t + 1, u] + log_probs[b, t, u, blank]
                    )
                cuda.atomic.add(lock, (b, u - 1), -1)
                t -= 1
            else:
                if cuda.atomic.add(lock, (b, u), 0) < 0:
                    if t == T[b] - 1:
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = (
                            beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        )
                    else:
                        # compute emission prob
                        emit = (
                            beta[b, t, u + 1] + log_probs[b, t, u, labels[b, u]]
                        )
                        # compute no_emission prob
                        no_emit = beta[b, t + 1, u] + log_probs[b, t, u, blank]
                        # do logsumexp between log_emit and log_no_emit
                        beta[b, t, u] = max(no_emit, emit) + math.log1p(
                            math.exp(-abs(no_emit - emit))
                        )
                    if u > 0:
                        cuda.atomic.add(lock, (b, u - 1), -1)
                    cuda.atomic.add(lock, (b, u), 1)
                    t -= 1
    if u == 0:
        # for each thread b (utterance)
        # normalize the loss over time
        log_p[b] = beta[b, 0, 0] / T[b]


@cuda.jit(
    "(float32[:,:,:,:], int32[:,:],float32[:,:,:], float32[:,:,:], float32[:,:,:,:], int32[:], int32[:], int32)"
)
def cu_kernel_compute_grad(log_probs, labels, alpha, beta, grads, T, U, blank):
    """
    Compute gradient for the forward-backward algorithm using Numba cuda kernel.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    Arguments
    ---------
    log_probs : tensor
        4D Tensor of (batch x TimeLength x LabelLength x outputDim) from the Transducer network.
    labels : tensor
        2D Tensor of (batch x MaxSeqLabelLength) containing targets of the batch with zero padding.
    beta : tensor
        3D Tensor of (batch x TimeLength x LabelLength) for backward computation.
    log_p : tensor
        1D Tensor of (batch) for backward cost computation.
    T : tensor
        1D Tensor of (batch) containing TimeLength of each target.
    U : tensor
        1D Tensor of (batch) containing LabelLength of each target.
    blank : int
        Blank indice.
    lock : int
        2D Tensor of (batch x LabelLength) containing bool(1-0) lock for parallel computation.
    """
    # parallelize the gradient computation over batch and timeseq length dim
    t = cuda.blockIdx.x
    b = cuda.threadIdx.x
    if t < T[b]:
        # compute the gradient for no_emit prob
        if t == 0:
            grads[b, T[b] - 1, U[b], blank] = -math.exp(
                alpha[b, T[b] - 1, U[b]]
                + log_probs[b, T[b] - 1, U[b], blank]
                - beta[b, 0, 0]
            )

        if t < T[b] - 1:
            for u in range(U[b] + 1):
                grads[b, t, u, blank] = alpha[b, t, u] + beta[b, t + 1, u]
                grads[b, t, u, blank] = -math.exp(
                    grads[b, t, u, blank]
                    + log_probs[b, t, u, blank]
                    - beta[b, 0, 0]
                )
        # compute the gradient for emit prob
        for u, l in enumerate(labels[b]):
            if u < U[b]:
                grads[b, t, u, l] = alpha[b, t, u] + beta[b, t, u + 1]
                grads[b, t, u, l] = -math.exp(
                    grads[b, t, u, l] + log_probs[b, t, u, l] - beta[b, 0, 0]
                )


class Transducer(PyLayer):
    """
    This class implements the Transducer loss computation with forward-backward algorithm
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    This class use torch.autograd.Function. In fact of using the forward-backward algorithm,
    we need to compute the gradient manually.

    This class can't be instantiated, please refer to TransducerLoss class

    It is also possible to use this class directly by using Transducer.apply
    """

    @staticmethod
    def forward(ctx, log_probs, labels, T, U, blank, reduction):
        log_probs = log_probs.detach()
        B, maxT, maxU, A = log_probs.shape
        grads = paddle.zeros(
            [B, maxT, maxU, A], dtype="float32",
        )
        alpha = paddle.zeros([B, maxT, maxU],)
        beta = paddle.zeros([B, maxT, maxU], )
        lock = paddle.zeros(
            [B, maxU], dtype=paddle.int32, 
        )
        log_p_alpha = paddle.zeros([B,], )
        log_p_beta = paddle.zeros([B,],)
        print("log_probs: {}".format(log_probs))
        print("labels: {}".format(labels))
        cu_kernel_forward[B, maxU](
            log_probs, labels, alpha, log_p_alpha, T, U, blank, lock,
        )
        lock = lock * 0
        cu_kernel_backward[B, maxU](
            log_probs, labels, beta, log_p_beta, T, U, blank, lock
        )
        cu_kernel_compute_grad[maxT, B](
            log_probs, labels, alpha, beta, grads, T, U, blank
        )
        ctx.grads = grads
        del alpha, beta, lock, log_p_beta, T, U, log_probs, labels
        paddle.cuda.empty_cache()
        if reduction == "mean":
            return -log_p_alpha.mean()
        elif reduction == "sum":
            return sum(-log_p_alpha)
        elif reduction == "none":
            return -log_p_alpha
        else:
            raise Exception("Unexpected reduction {}".format(reduction))

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.view(-1, 1, 1, 1).to(ctx.grads)
        return ctx.grads.mul_(grad_output), None, None, None, None, None, None


class TransducerLoss(Layer):
    """
    This class implements the Transduce loss computation with forward-backward algorithm.
    Sequence Transduction with naive implementation : https://arxiv.org/pdf/1211.3711.pdf

    The TranducerLoss(nn.Layer) use Transducer(autograd.Function)
    to compute the forward-backward loss and gradients.

    Example
    -------
    >>> import paddle
    >>> loss = TransducerLoss(blank=0)
    >>> acts = torch.randn((1,2,3,5)).cuda().log_softmax(dim=-1).requires_grad_()
    >>> labels = paddle.Tensor([[1,2]]).cuda().int()
    >>> act_length = paddle.Tensor([2]).cuda().int()
    >>> # U = label_length+1
    >>> label_length = paddle.Tensor([2]).cuda().int()
    >>> l = loss(acts, labels, act_length, label_length)
    >>> l.backward()
    """

    def __init__(self, blank=0, reduction="mean"):
        super(TransducerLoss, self).__init__()
        self.blank = blank
        self.reduction = reduction
        self.loss = Transducer.apply
        try:
            cuda.cuda_paths
        except ImportError:
            err_msg = "cannot import numba. To use Transducer loss\n"
            err_msg += "=============================\n"
            err_msg += "If you use your localhost:\n"
            err_msg += "pip install numba\n"
            err_msg += (
                "export NUMBAPRO_LIBDEVICE='/usr/local/cuda/nvvm/libdevice/' \n"
            )
            err_msg += "export NUMBAPRO_NVVM='/usr/local/cuda/nvvm/lib64/libnvvm.so' \n"
            err_msg += "================================ \n"
            err_msg += "If you use conda:\n"
            err_msg += "conda install numba cudatoolkit=9.0"
            raise ImportError(err_msg)

    def forward(self, log_probs, labels, T, U):
        return self.loss(log_probs, labels, T, U, self.blank, self.reduction)
