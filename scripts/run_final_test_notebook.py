"""Execute the final test notebook with saved outputs using installed IPython.

No nbformat/nbclient installation is needed. The notebook remains runnable in VS Code.
Outputs are saved after every cell, and training CSVLogger saves every epoch.
"""
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/05_final_test_evaluation.ipynb"


class Tee(io.StringIO):
    def __init__(self, console):
        super().__init__()
        self.console = console
    def write(self, value):
        result = super().write(value)
        self.console.write(value)
        self.console.flush()
        return result
    def flush(self):
        self.console.flush()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    shell = InteractiveShell.instance()
    namespace = {"__name__": "__main__"}
    os.chdir(ROOT / "notebooks")
    count = 0
    started = time.perf_counter()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        outputs = []
        def capture_display(*objects, **kwargs):
            for obj in objects:
                data, _ = shell.display_formatter.format(obj)
                outputs.append({"output_type": "display_data", "data": data, "metadata": {}})
        def capture_show(*args, **kwargs):
            for number in plt.get_fignums():
                figure = plt.figure(number)
                buffer = io.BytesIO()
                figure.savefig(buffer, format="png", bbox_inches="tight")
                outputs.append({"output_type": "display_data", "data": {
                    "image/png": base64.b64encode(buffer.getvalue()).decode(),
                    "text/plain": "<InceptionV3 final test figure>"}, "metadata": {}})
                plt.close(figure)
        namespace["display"] = capture_display
        plt.show = capture_show
        source = cell["source"].replace("from IPython.display import display, Markdown", "from IPython.display import Markdown")
        tee = Tee(sys.stdout)
        print(f"\n--- Executing code cell {count} ---", flush=True)
        error = None
        try:
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                exec(compile(source, f"{NOTEBOOK.name}:cell-{count}", "exec"), namespace)
        except Exception as exc:
            error = exc
            outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc),
                            "traceback": traceback.format_exc().splitlines()})
        if tee.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": tee.getvalue()})
        cell["execution_count"] = count
        cell["outputs"] = outputs
        notebook["metadata"]["evaluation_execution"] = {
            "method": "Sequential Python with captured IPython displays and Matplotlib outputs",
            "last_executed_code_cell": count, "completed": False,
            "elapsed_seconds": time.perf_counter() - started}
        NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if error:
            raise RuntimeError(f"Notebook stopped at code cell {count}: {error}") from error
        print(f"--- Completed code cell {count} ---", flush=True)
    notebook["metadata"]["evaluation_execution"]["completed"] = True
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("NOTEBOOK COMPLETE", flush=True)
    print(json.dumps(namespace["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
