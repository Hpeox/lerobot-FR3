"""Windows-only import aid for policy unit tests, NEVER imported by production.

This upstream fork eagerly imports POSIX-only Memmap converters. To test policy
interfaces on Windows without changing any upstream file, allow fcntl import but
make EVERY actual lock operation fail. No dataset converter/robot is tested here.
Linux uses the real fcntl module. This is not Windows deployment support.
"""

import importlib.util
import os
import sys
import types

WINDOWS_IMPORT_STUB = os.name == "nt" and importlib.util.find_spec("fcntl") is None

if WINDOWS_IMPORT_STUB:
    module = types.ModuleType("fcntl")

    def forbidden_os_operation(*args, **kwargs):
        raise RuntimeError("POSIX lock operation attempted in Windows policy-only tests")

    module.flock = forbidden_os_operation
    module.fcntl = forbidden_os_operation
    module.LOCK_EX, module.LOCK_NB, module.LOCK_UN, module.LOCK_SH = 2, 4, 8, 1
    sys.modules["fcntl"] = module
