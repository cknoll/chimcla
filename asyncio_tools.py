
import asyncio
from asyncio import run
from tqdm.asyncio import tqdm as tqdm_a


def background(f):
    """
    decorator for paralelization
    """
    # source: https://stackoverflow.com/a/59385935
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


async def main(func, arg_list):

    tasks = []
    for arg in arg_list:
        tasks.append(
            func(arg)
        )

    await tqdm_a.gather(*tasks)

    """
    # run this via:

    if __name__ == "__main__":


        aiot.run(aiot.main(myfunc, myargs))


    """