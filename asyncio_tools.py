
import asyncio
from asyncio import run
from tqdm.asyncio import tqdm as tqdm_a
import functools


def background(f):
    """
    decorator for paralelization
    """
    # source: https://stackoverflow.com/a/59385935
    def wrapped(arg, **kwargs):

        func_with_kwargs = functools.partial(f, **kwargs)
        return asyncio.get_event_loop().run_in_executor(None, func_with_kwargs, arg)

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