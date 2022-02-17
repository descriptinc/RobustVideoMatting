# Copyright (c) 2007–2014 Fabian Pedregosa.
# All rights reserved.
"""
Profile the memory usage of a Python program.
Adapted the script from https://github.com/pythonprofilers/memory_profiler/blob/ece5cb1b3be41c756336fad1432e5ab5f44ea193/memory_profiler.py
Added gpu memory profiling by following: https://github.com/wookayin/gpustat/blob/e70bc8565943dee15fd4338c176c821d7ab3d2eb/gpustat/core.py#L359
"""
import os
import subprocess
import sys
import time
import warnings


if sys.platform == "win32":
    # any value except signal.CTRL_C_EVENT and signal.CTRL_BREAK_EVENT
    # can be used to kill a process unconditionally in Windows
    SIGKILL = -1
else:
    from signal import SIGKILL
import psutil
import pynvml as N


# TODO: provide alternative when multiprocessing is not available
try:
    from multiprocessing import Process, Pipe
except ImportError:
    from multiprocessing.dummy import Process, Pipe

_TWO_20 = float(2**20)


# .. get available packages ..
try:
    import tracemalloc

    has_tracemalloc = True
except ImportError:
    has_tracemalloc = False


def _get_child_memory(process, meminfo_attr=None):
    """
    Returns a generator that yields memory for all child processes.
    """
    # Convert a pid to a process
    if isinstance(process, int):
        if process == -1:
            process = os.getpid()
        process = psutil.Process(process)

    if not meminfo_attr:
        # Use the psutil 2.0 attr if the older version isn't passed in.
        meminfo_attr = (
            "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        )

    # Select the psutil function get the children similar to how we selected
    # the memory_info attr (a change from excepting the AttributeError).
    children_attr = "children" if hasattr(process, "children") else "get_children"

    # Loop over the child processes and yield their memory
    try:
        for child in getattr(process, children_attr)(recursive=True):
            yield getattr(child, meminfo_attr)()[0] / _TWO_20
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # https://github.com/fabianp/memory_profiler/issues/71
        yield 0.0


def _get_memory(pid, backend, timestamps=False, include_children=False, filename=None):
    # .. low function to get memory consumption ..
    if pid == -1:
        pid = os.getpid()

    def tracemalloc_tool():
        # .. cross-platform but but requires Python 3.4 or higher ..
        stat = next(
            filter(
                lambda item: str(item).startswith(filename),
                tracemalloc.take_snapshot().statistics("filename"),
            )
        )
        mem = stat.size / _TWO_20
        if timestamps:
            return mem, time.time()
        else:
            return mem

    def ps_util_tool():
        # .. cross-platform but but requires psutil ..
        process = psutil.Process(pid)
        try:
            # avoid using get_memory_info since it does not exists
            # in psutil > 2.0 and accessing it will cause exception.
            meminfo_attr = (
                "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
            )
            mem = getattr(process, meminfo_attr)()[0] / _TWO_20
            if include_children:
                mem += sum(_get_child_memory(process, meminfo_attr))
            if timestamps:
                return mem, time.time()
            else:
                return mem
        except psutil.AccessDenied:
            pass
            # continue and try to get this from ps

    def posix_tool():
        # .. scary stuff ..
        if include_children:
            raise NotImplementedError(
                (
                    "The psutil module is required to monitor the "
                    "memory usage of child processes."
                )
            )

        warnings.warn("psutil module not found. memory_profiler will be slow")
        # ..
        # .. memory usage in MiB ..
        # .. this should work on both Mac and Linux ..
        # .. subprocess.check_output appeared in 2.7, using Popen ..
        # .. for backwards compatibility ..
        out = (
            subprocess.Popen(["ps", "v", "-p", str(pid)], stdout=subprocess.PIPE)
            .communicate()[0]
            .split(b"\n")
        )
        try:
            vsz_index = out[0].split().index(b"RSS")
            mem = float(out[1].split()[vsz_index]) / 1024
            if timestamps:
                return mem, time.time()
            else:
                return mem
        except:
            if timestamps:
                return -1, time.time()
            else:
                return -1

    if backend == "tracemalloc" and (filename is None or filename == "<unknown>"):
        raise RuntimeError("There is no access to source file of the profiled function")

    tools = {
        "tracemalloc": tracemalloc_tool,
        "psutil": ps_util_tool,
        "posix": posix_tool,
    }
    return tools[backend]()


class MemTimer(Process):
    """
    Fetch memory consumption from over a time interval
    """

    def __init__(
        self, monitor_pid, interval, pipe, backend, max_usage=False, *args, **kw
    ):
        self.monitor_pid = monitor_pid
        self.interval = interval
        self.pipe = pipe
        self.cont = True
        self.backend = backend
        self.max_usage = max_usage
        self.n_measurements = 1

        self.timestamps = kw.pop("timestamps", False)
        self.include_children = kw.pop("include_children", False)

        # get baseline memory usage
        self.mem_usage = [
            _get_memory(
                self.monitor_pid,
                self.backend,
                timestamps=self.timestamps,
                include_children=self.include_children,
            )
        ]
        self.gpu_index = kw.pop("gpu_index", None)
        self.gpu_mem_usage = None
        if self.gpu_index is not None:
            self.gpu_mem_usage = [_get_gpu_memory(self.gpu_index, self.monitor_pid)]

        super(MemTimer, self).__init__(*args, **kw)

    def run(self):
        self.pipe.send(0)  # we're ready
        stop = False
        while True:
            cur_mem = _get_memory(
                self.monitor_pid,
                self.backend,
                timestamps=self.timestamps,
                include_children=self.include_children,
            )

            if not self.max_usage:
                self.mem_usage.append(cur_mem)
            else:
                self.mem_usage[0] = max(cur_mem, self.mem_usage[0])
            if self.gpu_index is not None:
                curr_gpu_mem = _get_gpu_memory(self.gpu_index, self.monitor_pid)
                if not self.max_usage:
                    self.gpu_mem_usage.append(curr_gpu_mem)
                else:
                    self.gpu_mem_usage[0] = max(curr_gpu_mem, self.gpu_mem_usage[0])
            self.n_measurements += 1
            if stop:
                break
            stop = self.pipe.poll(self.interval)
            # do one more iteration

        self.pipe.send(self.mem_usage)
        self.pipe.send(self.n_measurements)
        if self.gpu_index is not None:
            self.pipe.send(self.gpu_mem_usage)


def _get_gpu_memory(index, pid):
    """
    Memory utilized by pid in MBs.

    Due to container isolation, the pid supplied inside container doesn't
    match the one on host. Thus the same process will have different pid inside
    and outside the container. The nvml API is only aware of the host-pid.
    Since it's non-trivial to obtain this mapping due to isolation, our only
    recourse is to monitor the whole GPU and ensure that only the monitored
    process is using the specified GPU.
    """
    N.nvmlInit()

    handle = N.nvmlDeviceGetHandleByIndex(index)

    try:
        memory = N.nvmlDeviceGetMemoryInfo(handle)  # in bytes
    except N.NVMLError:
        memory = None
    return memory.used / 1024**2 if memory else 0

    # Uncomment below code when running on host.
    # It's not possible to query nvml using pid inside the container.

    # try:
    #     nv_comp_processes = N.nvmlDeviceGetComputeRunningProcesses(handle)
    # except N.NVMLError:
    #     nv_comp_processes = []  # Not supported
    # try:
    #     nv_graphics_processes = N.nvmlDeviceGetGraphicsRunningProcesses(handle)
    # except N.NVMLError:
    #     nv_graphics_processes = []  # Not supported
    # procs = nv_graphics_processes + nv_comp_processes
    # for proc in procs:
    #     if proc.pid == pid:
    #         return proc.usedGpuMemory // 1024 * 1024 if proc.usedGpuMemory else 0
    # return 0


def memory_usage(
    proc=-1,
    interval=0.1,
    timeout=None,
    timestamps=False,
    include_children=False,
    multiprocess=False,
    max_usage=False,
    retval=False,
    stream=None,
    backend=None,
    max_iterations=None,
    gpu_index=None,
):
    """
    Return the memory usage of a process or piece of code

    Parameters
    ----------
    proc : {int, string, tuple, subprocess.Popen}, optional
        The process to monitor. Can be given by an integer/string
        representing a PID, by a Popen object or by a tuple
        representing a Python function. The tuple contains three
        values (f, args, kw) and specifies to run the function
        f(*args, **kw).
        Set to -1 (default) for current process.

    interval : float, optional
        Interval at which measurements are collected.

    timeout : float, optional
        Maximum amount of time (in seconds) to wait before returning.

    max_usage : bool, optional
        Only return the maximum memory usage (default False)

    retval : bool, optional
        For profiling python functions. Save the return value of the profiled
        function. Return value of memory_usage becomes a tuple:
        (mem_usage, retval)

    timestamps : bool, optional
        if True, timestamps of memory usage measurement are collected as well.

    include_children : bool, optional
        if True, sum the memory of all forked processes as well

    multiprocess : bool, optional
        if True, track the memory usage of all forked processes.

    stream : File
        if stream is a File opened with write access, then results are written
        to this file instead of stored in memory and returned at the end of
        the subprocess. Useful for long-running processes.
        Implies timestamps=True.

    max_iterations : int
        Limits the number of iterations (calls to the process being monitored). Relevent
        when the process is a python function.

    Returns
    -------
    mem_usage : list of floating-point values
        memory usage, in MiB. It's length is always < timeout / interval
        if max_usage is given, returns the two elements maximum memory and
        number of measurements effectuated
    ret : return value of the profiled function
        Only returned if retval is set to True
    """
    backend = choose_backend(backend)
    if stream is not None:
        timestamps = True

    if not max_usage:
        ret = []
    else:
        ret = -1

    if timeout is not None:
        max_iter = int(round(timeout / interval))
    elif isinstance(proc, int):
        # external process and no timeout
        max_iter = 1
    else:
        # for a Python function wait until it finishes
        max_iter = float("inf")
        if max_iterations is not None:
            max_iter = max_iterations

    if callable(proc):
        proc = (proc, (), {})
    if isinstance(proc, (list, tuple)):
        if len(proc) == 1:
            f, args, kw = (proc[0], (), {})
        elif len(proc) == 2:
            f, args, kw = (proc[0], proc[1], {})
        elif len(proc) == 3:
            f, args, kw = (proc[0], proc[1], proc[2])
        else:
            raise ValueError

        current_iter = 0
        while True:
            current_iter += 1
            child_conn, parent_conn = Pipe()  # this will store MemTimer's results
            p = MemTimer(
                os.getpid(),
                interval,
                child_conn,
                backend,
                timestamps=timestamps,
                max_usage=max_usage,
                include_children=include_children,
                gpu_index=gpu_index,
            )
            p.start()
            parent_conn.recv()  # wait until we start getting memory

            # When there is an exception in the "proc" - the (spawned) monitoring processes don't get killed.
            # Therefore, the whole process hangs indefinitely. Here, we are ensuring that the process gets killed!
            try:
                returned = f(*args, **kw)
                parent_conn.send(0)  # finish timing
                mem = parent_conn.recv()
                n_measurements = parent_conn.recv()
                gpu_mem = parent_conn.recv() if gpu_index is not None else [0]
                ret = mem, gpu_mem
                if max_usage:
                    # Convert the one element list produced by MemTimer to a singular value
                    ret = (mem[0], gpu_mem[0])
                if retval:
                    ret = ret, returned
            except Exception:
                parent = psutil.Process(os.getpid())
                for child in parent.children(recursive=True):
                    os.kill(child.pid, SIGKILL)
                p.join(0)
                raise

            p.join(5 * interval)

            if (n_measurements > 4) or (current_iter == max_iter) or (interval < 1e-6):
                break
            interval /= 10.0
    elif isinstance(proc, subprocess.Popen):
        # external process, launched from Python
        line_count = 0
        while True:
            if not max_usage:
                mem_usage = _get_memory(
                    proc.pid,
                    backend,
                    timestamps=timestamps,
                    include_children=include_children,
                )

                if stream is not None:
                    stream.write("MEM {0:.6f} {1:.4f}\n".format(*mem_usage))

                    # Write children to the stream file
                    if multiprocess:
                        for idx, chldmem in enumerate(_get_child_memory(proc.pid)):
                            stream.write(
                                "CHLD {0} {1:.6f} {2:.4f}\n".format(
                                    idx, chldmem, time.time()
                                )
                            )
                else:
                    # Create a nested list with the child memory
                    if multiprocess:
                        mem_usage = [mem_usage]
                        for chldmem in _get_child_memory(proc.pid):
                            mem_usage.append(chldmem)

                    # Append the memory usage to the return value
                    ret.append(mem_usage)
            else:
                ret = max(
                    ret,
                    _get_memory(proc.pid, backend, include_children=include_children),
                )
            time.sleep(interval)
            line_count += 1
            # flush every 50 lines. Make 'tail -f' usable on profile file
            if line_count > 50:
                line_count = 0
                if stream is not None:
                    stream.flush()
            if timeout is not None:
                max_iter -= 1
                if max_iter == 0:
                    break
            if proc.poll() is not None:
                break
    else:
        # external process
        if max_iter == -1:
            max_iter = 1
        counter = 0
        while counter < max_iter:
            counter += 1
            if not max_usage:
                mem_usage = _get_memory(
                    proc,
                    backend,
                    timestamps=timestamps,
                    include_children=include_children,
                )
                if stream is not None:
                    stream.write("MEM {0:.6f} {1:.4f}\n".format(*mem_usage))

                    # Write children to the stream file
                    if multiprocess:
                        for idx, chldmem in enumerate(_get_child_memory(proc)):
                            stream.write(
                                "CHLD {0} {1:.6f} {2:.4f}\n".format(
                                    idx, chldmem, time.time()
                                )
                            )
                else:
                    # Create a nested list with the child memory
                    if multiprocess:
                        mem_usage = [mem_usage]
                        for chldmem in _get_child_memory(proc):
                            mem_usage.append(chldmem)

                    # Append the memory usage to the return value
                    ret.append(mem_usage)
            else:
                ret = max(
                    [ret, _get_memory(proc, backend, include_children=include_children)]
                )

            time.sleep(interval)
            # Flush every 50 lines.
            if counter % 50 == 0 and stream is not None:
                stream.flush()
    if stream:
        return None
    return ret


def choose_backend(new_backend=None):
    """
    Function that tries to setup backend, chosen by user, and if failed,
    setup one of the allowable backends
    """

    _backend = "no_backend"
    all_backends = [
        ("psutil", True),
        ("posix", os.name == "posix"),
        ("tracemalloc", has_tracemalloc),
    ]
    backends_indices = dict((b[0], i) for i, b in enumerate(all_backends))

    if new_backend is not None:
        all_backends.insert(0, all_backends.pop(backends_indices[new_backend]))

    for n_backend, is_available in all_backends:
        if is_available:
            _backend = n_backend
            break
    if _backend != new_backend and new_backend is not None:
        warnings.warn(
            "{0} can not be used, {1} used instead".format(new_backend, _backend)
        )
    return _backend
