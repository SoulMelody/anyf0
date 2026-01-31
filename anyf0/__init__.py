import contextlib
import os

with contextlib.suppress(ImportError):
    from nvidia import cublas, cuda_runtime, cudnn, cufft, nvjitlink
    os.environ["PATH"] += os.pathsep + os.sep.join([cuda_runtime.__path__[0], "bin"])
    os.environ["PATH"] += os.pathsep + os.sep.join([cublas.__path__[0], "bin"])
    os.environ["PATH"] += os.pathsep + os.sep.join([cudnn.__path__[0], "bin"])
    os.environ["PATH"] += os.pathsep + os.sep.join([cufft.__path__[0], "bin"])
    os.environ["PATH"] += os.pathsep + os.sep.join([nvjitlink.__path__[0], "bin"])