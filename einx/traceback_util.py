import threading
import os
import traceback
import types
import functools

path = os.path.abspath(os.path.join(__file__, ".."))


def include_frame(fname):
    return not fname.startswith(path)


thread_local = threading.local()


def _set_in_reraise():
    if not hasattr(thread_local, "in_reraise"):
        thread_local.in_reraise = False
    assert not thread_local.in_reraise
    thread_local.in_reraise = True


def _unset_in_reraise():
    assert thread_local.in_reraise
    thread_local.in_reraise = False


def _is_in_reraise():
    return getattr(thread_local, "in_reraise", False)


def _filter_tb(tb):
    tb_list = list(traceback.walk_tb(tb))
    first_excluded_idx = 0
    while first_excluded_idx < len(tb_list) and include_frame(
        tb_list[first_excluded_idx][0].f_code.co_filename
    ):
        first_excluded_idx += 1
    last_excluded_idx = len(tb_list) - 1
    while last_excluded_idx >= 0 and include_frame(
        tb_list[last_excluded_idx][0].f_code.co_filename
    ):
        last_excluded_idx -= 1

    if first_excluded_idx <= last_excluded_idx:
        tb_list1 = tb_list[:first_excluded_idx]
        tb_list2 = tb_list[last_excluded_idx + 1 :]
        tb_list = tb_list1 + tb_list2
        tb = None
        for f, line_no in tb_list:
            tb = types.TracebackType(tb, f, f.f_lasti, line_no)

    return tb


def filter(func):
    filter = os.environ.get("EINX_FILTER_TRACEBACK", "true").lower() in ("true", "yes", "1")

    if filter:

        @functools.wraps(func)
        def func_with_reraise(*args, **kwargs):
            if not _is_in_reraise():
                _set_in_reraise()
                tb = None
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tb = _filter_tb(e.__traceback__)
                    raise e.with_traceback(tb) from None
                finally:
                    del tb
                    _unset_in_reraise()
            else:
                return func(*args, **kwargs)

        return func_with_reraise
    else:
        return func
