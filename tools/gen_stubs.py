"""Regenerate the type stubs of the C++ extension (modcma/c_maes/cmaescpp/*.pyi) with
nanobind.stubgen, from the installed extension:

    pip install --no-build-isolation -e .
    python tools/gen_stubs.py

Commit the result together with every change of the bindings (bindings/*.cpp).
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "modcma", "c_maes", "cmaescpp")


def main():
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(
            [sys.executable, "-m", "nanobind.stubgen", "-m", "modcma.c_maes.cmaescpp", "-r", "-O", out, "-q"],
            check=True,
        )
        generated = os.path.join(out, "cmaescpp")
        for name in os.listdir(TARGET):
            if name.endswith(".pyi"):
                os.remove(os.path.join(TARGET, name))
        for name in sorted(os.listdir(generated)):
            shutil.copy(os.path.join(generated, name), os.path.join(TARGET, name))
            print(f"wrote {os.path.relpath(os.path.join(TARGET, name), ROOT)}")


if __name__ == "__main__":
    main()
