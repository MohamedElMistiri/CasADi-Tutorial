/* Start Python only after an explicit request. Saved results need no kernel. */
(() => {
  "use strict";
  const payload = document.getElementById("casadi-notebook");
  if (!payload) return;
  const notebook = JSON.parse(payload.textContent);
  const toolbar = document.querySelector(".casadi-session");
  const start = toolbar.querySelector("[data-casadi-start]");
  const run = toolbar.querySelector("[data-casadi-run]");
  const stop = toolbar.querySelector("[data-casadi-stop]");
  const status = toolbar.querySelector("[data-casadi-status]");
  const cells = [...document.querySelectorAll("[data-casadi-cell]")];
  let session = null;
  let liveNotebook = null;
  let busy = false;
  let library = null;

  function loadThebe() {
    if (window.thebe) return Promise.resolve();
    if (library) return library;
    library = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://unpkg.com/thebe@0.9.2/lib/index.js";
      script.onload = resolve;
      script.onerror = () => {
        library = null;
        script.remove();
        reject(new Error("Could not download the interactive tools."));
      };
      document.head.append(script);
    });
    return library;
  }

  function setBusy(value) {
    busy = value;
    start.disabled = run.disabled = stop.disabled = value;
    toolbar.setAttribute("aria-busy", String(value));
  }

  function restoreSaved() {
    document.querySelectorAll(".casadi-live-cell").forEach(element => element.remove());
    document.querySelectorAll("[data-casadi-input], [data-casadi-saved]").forEach(element => {
      element.hidden = false;
    });
  }

  async function disconnect() {
    if (session) {
      try { await session.shutdown(); } catch (_) { /* Already expired. */ }
    }
    session = liveNotebook = null;
    restoreSaved();
    start.hidden = false;
    start.textContent = "Enable interactive session";
    run.hidden = stop.hidden = true;
    status.textContent = "Session disconnected. Saved results are shown below.";
  }

  async function runCells() {
    for (let index = 0; index < liveNotebook.cells.length; index++) {
      status.textContent = `Running cell ${index + 1} of ${liveNotebook.cells.length}…`;
      const cell = liveNotebook.cells[index];
      const result = await cell.execute(cell.source);
      if (result?.error || cell.outputs.some(output => output.output_type === "error")) {
        throw new Error(`Cell ${index + 1} needs attention. Inspect its output, edit it, and retry.`);
      }
      cells[index].querySelectorAll("[data-casadi-saved]").forEach(output => {
        output.hidden = true;
      });
    }
    status.textContent = "Session ready. Edit a code cell or use the experiment controls below.";
  }

  function configuration() {
    const config = {
      useBinder: true,
      requestKernel: true,
      // The page already uses MathJax 3. Avoid Thebe's legacy MathJax 2 loader.
      mathjaxUrl: "",
      mountActivateWidget: false,
      mountStatusWidget: false,
      mountRunButton: true,
      mountRunAllButton: false,
      mountRestartButton: false,
      mountRestartAllButton: false,
      selector: ".casadi-live-cell [data-executable]",
      binderOptions: {repo: notebook.repo, ref: notebook.ref},
      kernelOptions: {kernelName: "python3", path: notebook.path.split("/").slice(0, -1).join("/")},
      savedSessionOptions: {enabled: true, maxAge: 3600, storagePrefix: "casadi-course"},
    };
    // The local preview injects these temporary settings in memory only.
    // Production pages always use Binder; they contain no server credentials.
    const local = window.casadiLocalRuntime;
    if (["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) && local) {
      const endpoint = new URL(local.baseUrl);
      if (!["localhost", "127.0.0.1", "[::1]"].includes(endpoint.hostname)) {
        throw new Error("The local runtime must use a loopback address.");
      }
      config.useBinder = false;
      config.serverSettings = {baseUrl: local.baseUrl, token: local.token};
      config.kernelOptions.kernelName = local.kernelName || "python3";
      config.savedSessionOptions.enabled = false;
    }
    return config;
  }

  start.addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    status.textContent = "Starting Python… You can keep reading the saved results while you wait.";
    try {
      await loadThebe();
      restoreSaved();
      cells.forEach((cell, index) => {
        const live = document.createElement("div");
        live.className = "casadi-live-cell";
        const pre = document.createElement("pre");
        pre.dataset.executable = "";
        pre.dataset.language = "python";
        pre.textContent = notebook.code[index];
        live.append(pre);
        const input = cell.querySelector("[data-casadi-input]");
        input.hidden = true;
        input.after(live);
        cell.querySelectorAll(":scope > .jp-Cell-outputWrapper").forEach(output => {
          output.dataset.casadiSaved = "";
        });
      });
      const connected = await window.thebe.bootstrap(configuration());
      if (!connected.session) throw new Error("Python could not be started. Please try again.");
      session = connected.session;
      liveNotebook = connected.notebook;
      start.hidden = true;
      run.hidden = stop.hidden = false;
      if (notebook.path.startsWith("exercises/")) {
        status.textContent = "Session ready. Complete and run exercise cells individually.";
      } else {
        await runCells();
      }
    } catch (error) {
      status.textContent = `${error.message} Saved results remain available. You can also open JupyterLab.`;
      if (!session) {
        restoreSaved();
        start.textContent = "Retry interactive session";
      }
    } finally {
      setBusy(false);
    }
  });

  run.addEventListener("click", async () => {
    if (busy || !liveNotebook) return;
    setBusy(true);
    try { await runCells(); }
    catch (error) { status.textContent = error.message; }
    finally { setBusy(false); }
  });
  stop.addEventListener("click", async () => {
    if (busy) return;
    setBusy(true);
    try { await disconnect(); }
    finally { setBusy(false); }
  });
})();
