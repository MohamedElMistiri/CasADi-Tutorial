"""Serve a built personal website with a temporary local Jupyter runtime.

    python scripts/serve_website_preview.py --site-dir build/personal-site-source/_site

The preview binds to loopback only. Temporary runtime credentials are injected
into responses in memory; production HTML files remain unchanged. Use Ctrl+C
to stop both the preview and its kernels. Requires the course environment.
"""

import argparse
import functools
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument(
        "--tutorial-dir", type=Path, help="Serve a separate course build at /tutorials/casadi/"
    )
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--kernel-port", type=int, default=8875)
    args = parser.parse_args()
    site = args.site_dir.resolve()
    if not (site / "index.html").is_file():
        parser.error("--site-dir must contain a built website with index.html")
    base = f"http://127.0.0.1:{args.kernel_port}/"
    origin = f"http://127.0.0.1:{args.port}"
    token = secrets.token_hex(24)
    runtime = ROOT / "build" / ("preview-runtime-" + secrets.token_hex(4))
    data = runtime / "jupyter"
    spec = data / "kernels/casadi-preview"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Local CasADi preview",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        JUPYTER_PATH=os.pathsep.join([str(data), str(Path(sys.base_prefix) / "share/jupyter")]),
        JUPYTER_CONFIG_DIR=str(runtime / "config"),
        JUPYTER_DATA_DIR=str(data),
        JUPYTER_RUNTIME_DIR=str(runtime / "run"),
        IPYTHONDIR=str(runtime / "ipython"),
        PYTHONPATH=str(ROOT),
        CASADI_TUTORIAL_WIDGETS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MPLBACKEND="module://matplotlib_inline.backend_inline",
    )
    env.pop("CASADI_TUTORIAL_PLOTS", None)
    settings = json.dumps({"baseUrl": base, "token": token, "kernelName": "casadi-preview"})
    injection = f"<script>window.casadiLocalRuntime={settings};</script>"

    class PreviewHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            path = unquote(urlsplit(path).path)
            prefix = "/tutorials/casadi/"
            if args.tutorial_dir and path.startswith(prefix):
                course = args.tutorial_dir.resolve()
                target = (course / path.removeprefix(prefix)).resolve()
                return str(target if target.is_relative_to(course) else course / "__not_found__")
            return super().translate_path(path)

        def do_GET(self):
            path = Path(self.translate_path(unquote(urlsplit(self.path).path)))
            if path.is_dir():
                path /= "index.html"
            if path.is_file() and path.suffix == ".html":
                # translate_path already confines requests to the served directory.
                body = path.read_text(encoding="utf-8").replace("<head>", "<head>" + injection, 1)
                body = body.replace("http://127.0.0.1:8080/", origin + "/")
                encoded = body.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)
            else:
                super().do_GET()

        def log_message(self, *args):
            pass

    log = (runtime / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter_server",
            "--no-browser",
            f"--ServerApp.root_dir={ROOT}",
            f"--ServerApp.port={args.kernel_port}",
            "--ServerApp.port_retries=0",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.allow_origin={origin}",
            "--ServerApp.ip=127.0.0.1",
        ],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    server = None
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError(f"Jupyter exited. Inspect {runtime / 'server.log'}")
            try:
                req = urllib.request.Request(
                    base + "api", headers={"Authorization": "token " + token}
                )
                urllib.request.urlopen(req, timeout=1).close()
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Jupyter did not become ready")

        # Windows permits overlapping binds with SO_REUSEADDR, which can route
        # requests to a different existing preview. Fail instead of sharing it.
        class PreviewServer(ThreadingHTTPServer):
            allow_reuse_address = False

        server = PreviewServer(
            ("127.0.0.1", args.port), functools.partial(PreviewHandler, directory=str(site))
        )
        print(f"Personal website: {origin}/", flush=True)
        print(f"CasADi tutorial:  {origin}/tutorials/casadi/", flush=True)
        print("Interactive sessions use local Python. Press Ctrl+C to stop.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        try:
            req = urllib.request.Request(
                base + "api/shutdown", method="POST", headers={"Authorization": "token " + token}
            )
            urllib.request.urlopen(req, timeout=5).close()
            process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=10)
        log.close()


if __name__ == "__main__":
    main()
