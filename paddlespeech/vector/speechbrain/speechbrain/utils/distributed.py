"""Guard for running certain operations on main process only

Authors:
 * Abdel Heba 2020
 * Aku Rouhe 2020
"""
import os
import paddle
import logging
import paddle.distributed as dist

logger = logging.getLogger(__name__)


def run_on_main(
    func,
    args=None,
    kwargs=None,
    post_func=None,
    post_args=None,
    post_kwargs=None,
    run_post_on_main=False,
):
    """Runs a function with DPP (multi-gpu) support.

    The main function is only run on the main process.
    A post_function can be specified, to be on non-main processes after the main
    func completes. This way whatever the main func produces can be loaded on
    the other processes.

    Arguments
    ---------
    func : callable
        Function to run on the main process.
    args : list, None
        Positional args to pass to func.
    kwargs : dict, None
        Keyword args to pass to func.
    post_func : callable, None
        Function to run after func has finished on main. By default only run on
        non-main processes.
    post_args : list, None
        Positional args to pass to post_func.
    post_kwargs : dict, None
        Keyword args to pass to post_func.
    run_post_on_main : bool
        Whether to run post_func on main process as well. (default: False)
    """
    # Handle the mutable data types' default args:
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    if post_args is None:
        post_args = []
    if post_kwargs is None:
        post_kwargs = {}

    if if_main_process():
        # Main comes here
        try:
            func(*args, **kwargs)
        finally:
            ddp_barrier()
    else:
        # Others go here
        ddp_barrier()
    if post_func is not None:
        if run_post_on_main:
            # Just run on every process without any barrier.
            post_func(*post_args, **post_kwargs)
        elif not if_main_process():
            # Others go here
            try:
                post_func(*post_args, **post_kwargs)
            finally:
                ddp_barrier()
        else:
            # But main comes here
            ddp_barrier()


def if_main_process():
    """Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.
    """
    # todo 如何判断是main process
    return dist.get_rank() == 0
    # if "RANK" in os.environ:
    #     if os.environ["RANK"] == "":
    #         return False
    #     else:
    #         print(os.environ["RANK"])
    #         if int(os.environ["RANK"]) == 0:
    #             return True
    #         return False
    # return True


def ddp_barrier():
    """In DDP mode, this function will synchronize all processes.
    torch.distributed.barrier() will block processes until the whole
    group enters this function.
    """
    # if torch.distributed.is_initialized():
    if dist.get_rank() > 1:
        paddle.distributed.barrier()


def ddp_init_group(run_opts):
    """This function will initialize the ddp group if
    distributed_launch=True bool is given in the python command line.

    The ddp group will use distributed_backend arg for setting the
    DDP communication protocol. `RANK` Unix variable will be used for
    registering the subprocess to the ddp group.

    Arguments
    ---------
    run_opts: list
        A list of arguments to parse, most often from `sys.argv[1:]`.
    """
    if run_opts["distributed_launch"]:
        if "local_rank" not in run_opts:
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"
            )
        else:
            if run_opts["local_rank"] + 1 > torch.cuda.device_count():
                raise ValueError(
                    "Killing process " + str() + "\n"
                    "Not enough GPUs available!"
                )
        if "RANK" in os.environ is None or os.environ["RANK"] == "":
            raise ValueError(
                "To use DDP backend, start your script with:\n\t"
                "python -m torch.distributed.launch [args]\n\t"
                "experiment.py hyperparams.yaml --distributed_launch=True "
                "--distributed_backend=nccl"
            )
        rank = int(os.environ["RANK"])

        if run_opts["distributed_backend"] == "nccl":
            if not torch.distributed.is_nccl_available():
                raise ValueError("NCCL is not supported in your machine.")
        elif run_opts["distributed_backend"] == "gloo":
            if not torch.distributed.is_gloo_available():
                raise ValueError("GLOO is not supported in your machine.")
        elif run_opts["distributed_backend"] == "mpi":
            if not torch.distributed.is_mpi_available():
                raise ValueError("MPI is not supported in your machine.")
        else:
            logger.info(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
            raise ValueError(
                run_opts["distributed_backend"]
                + " communcation protocol doesn't exist."
            )
        # rank arg is used to set the right rank of the current process for ddp.
        # if you have 2 servers with 2 gpu:
        # server1:
        #   GPU0: local_rank=device=0, rank=0
        #   GPU1: local_rank=device=1, rank=1
        # server2:
        #   GPU0: local_rank=device=0, rank=2
        #   GPU1: local_rank=device=1, rank=3
        torch.distributed.init_process_group(
            backend=run_opts["distributed_backend"], rank=rank
        )
    else:
        logger.info(
            "distributed_launch flag is disabled, "
            "this experiment will be executed without DDP."
        )
        if "local_rank" in run_opts and run_opts["local_rank"] > 0:
            raise ValueError(
                "DDP is disabled, local_rank must not be set.\n"
                "For DDP training, please use --distributed_launch=True. "
                "For example:\n\tpython -m torch.distributed.launch "
                "experiment.py hyperparams.yaml "
                "--distributed_launch=True --distributed_backend=nccl"
            )
