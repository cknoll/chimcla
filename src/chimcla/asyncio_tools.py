"""
Helper module; provides useful functions for running python code in background or in parallel.
"""

import asyncio
from asyncio import run
from tqdm.asyncio import tqdm as tqdm_a
import functools


def background(f):
    """
    decorator for parallelization
    """
    # source: https://stackoverflow.com/a/59385935
    from ipydex import IPS
    def wrapped(arg, arg2=None, **kwargs):
        if arg2:
            print(f"{arg=} ({type(arg)=}), {arg2=} ({type(arg)=})")

        func_with_kwargs = functools.partial(f, **kwargs)
        return asyncio.get_event_loop().run_in_executor(None, func_with_kwargs, arg)
    #
    # IPS()

    return wrapped


async def main(func, arg_list, **kwargs):

    tasks = []
    for arg in arg_list:
        tasks.append(
            func(arg, **kwargs)
        )

    await tqdm_a.gather(*tasks)

    """
    # run this via:

    if __name__ == "__main__":


        aiot.run(aiot.main(myfunc, myargs))

    """

async_main = main

def async_run(func, arg_list, **kwargs):
    run(async_main(func, arg_list, **kwargs))
