"""Make b12x and quack import on Windows: fall back to a msvcrt-based flock.

Usage: python patch_fcntl.py <site-packages dir>
Idempotent; rerun after reinstalling either package.
"""
import shutil
import sys
from pathlib import Path

site = Path(sys.argv[1])
shim = Path(__file__).with_name("winflock.py")

# package -> (shim module path inside the package, files importing fcntl)
TARGETS = {
    "b12x": ("_lib/winflock.py", ["preparation/_cache.py", "preparation/artifacts.py"]),
    "quack": ("cache/_winflock.py", ["cache/jit.py", "cache/async_compile.py"]),
}

def patch_b12x_future_kernel() -> None:
    """With program retention active, b12x's Triton run() hook calls
    program_keys() on every result, including the FutureKernel that Triton's
    async compile (used by vLLM's JIT warmup) returns. Skip it: the compiled
    kernel is recorded when it is later launched."""
    p = site / "b12x" / "_lib" / "compile_plan.py"
    if not p.exists():
        return
    t = p.read_text(encoding="utf-8")
    old = ('    raise TypeError(f"compile factory returned an unannotated '
           '{type(value).__name__}")\n')
    new = ('    if type(value).__name__ == "FutureKernel":  # Triton async compile\n'
           '        return ()\n' + old)
    if "Triton async compile" in t:
        print("already patched b12x _lib/compile_plan.py")
        return
    assert t.count(old) == 1
    backup = p.with_name(p.name + ".orig")
    if not backup.exists():
        shutil.copyfile(p, backup)
    p.write_text(t.replace(old, new, 1), encoding="utf-8")
    print("patched b12x _lib/compile_plan.py")


patch_b12x_future_kernel()

for pkg, (shim_rel, files) in TARGETS.items():
    root = site / pkg
    if not root.is_dir():
        print("not installed:", pkg)
        continue
    shutil.copyfile(shim, root / shim_rel)
    module = f"{pkg}.{shim_rel[:-3].replace('/', '.')}"
    mod_pkg, _, mod_name = module.rpartition(".")
    new = ("try:\n    import fcntl\nexcept ImportError:  # Windows\n"
           f"    from {mod_pkg} import {mod_name} as fcntl\n")
    for rel in files:
        p = root / rel
        t = p.read_text(encoding="utf-8")
        if "winflock" in t:
            print("already patched", pkg, rel)
            continue
        assert t.count("import fcntl\n") == 1, (pkg, rel)
        backup = p.with_name(p.name + ".orig")
        if not backup.exists():
            shutil.copyfile(p, backup)
        p.write_text(t.replace("import fcntl\n", new, 1), encoding="utf-8")
        print("patched", pkg, rel)
