"""
Private testing helpers for the polars-bigquery client.

IMPORTANT: Rationale for this module's location
-----------------------------------------------
This module is placed inside the production package (`polars_bigquery`) rather than
the `tests/` directory to bypass pytest's automatic frame instrumentation.

Pytest automatically rewrites assertions and instruments frames for all Python files
located under the `tests/` directory. This instrumentation keeps hidden references
to local variables alive in pytest's internal traceback and local variables cache
so it can print them on failure. 

For tests that verify garbage collection and destructors (like checking if Rust's 
`Drop` is triggered to abort background tasks), this instrumentation leaks references
and prevents objects from ever being deallocated during the test execution.

By placing these GC-sensitive helper functions in this production-packaged module,
they run in clean, standard CPython frames that are completely untouched by pytest.
This allows `del` and `gc.collect()` to work deterministically, ensuring that
refcounts drop to 0 and trigger the Rust destructors during the test.

Furthermore, we use a mutable state object and nested phase functions to completely
avoid CPython 3.12+ stack/register caching of local variables in the outer helper
frame. By storing the GC-sensitive objects (exporter, capsule) only as attributes 
on a state object or as locals in short-lived nested frames, we ensure they are 
deterministically deallocated the moment we clear them, without being kept alive
on the active outer frame's stack during polling.
"""

import gc
import time
from polars_bigquery import polars_bigquery

class TestState:
    """A mutable container to hold GC-sensitive objects without caching them on the stack."""
    def __init__(self):
        self.exporter = None
        self.capsule = None
        self.drop_flag = None


def run_exporter_drop_test():
    state = TestState()

    def phase_create():
        res = polars_bigquery._test_create_exporter_with_drop_flag()
        state.exporter = res[0]
        state.drop_flag = res[1]
        assert not state.drop_flag.is_set()
        # res goes out of scope here

    phase_create()
    gc.collect()

    # Clear the exporter reference. Since it is not a local variable in this frame,
    # there are no stack/register references keeping it alive.
    state.exporter = None
    gc.collect()
    
    # Poll the flag
    success = False
    for _ in range(20):
        if state.drop_flag.is_set():
            success = True
            break
        time.sleep(0.05)
        
    return success


def run_exporter_drop_after_stream_created_test():
    state = TestState()

    def phase_create_and_export():
        res = polars_bigquery._test_create_exporter_with_drop_flag()
        state.exporter = res[0]
        state.drop_flag = res[1]
        assert not state.drop_flag.is_set()
        
        # Create capsule (moves receiver out of exporter)
        state.capsule = state.exporter.__arrow_c_stream__()
        assert not state.drop_flag.is_set()
        # res goes out of scope here

    phase_create_and_export()
    gc.collect()
    
    # Drop the exporter. The task should still run because capsule owns the receiver.
    state.exporter = None
    gc.collect()
    if state.drop_flag.is_set():
        return False
        
    # Drop the capsule. This should trigger the FFI release callback and abort the task.
    state.capsule = None
    gc.collect()
    
    # Poll the flag
    success = False
    for _ in range(20):
        if state.drop_flag.is_set():
            success = True
            break
        time.sleep(0.05)
        
    return success
