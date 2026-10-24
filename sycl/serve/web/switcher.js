// The model menu (sycl/serve/server_intel.py): a host that swaps models on one GPU (config model_switcher) offers
// the ones served on this port. Switching stops THIS server and starts the other on the same port, so the page
// waits for /health to name the new model, then reloads. Uses app.js's headers().
(() => {
  const box = document.createElement("label");
  box.className = "model-switch";
  box.hidden = true;
  box.title = "Which model is loaded. Only one fits on the GPU at a time: switching unloads this one and loads the other (about 2 minutes).";
  box.innerHTML = '<span class="model-switch__k">Model</span><select aria-label="Model on the GPU"></select>' +
                  '<span class="model-switch__s" role="status"></span>';
  const sel = box.querySelector("select"), status = box.querySelector(".model-switch__s");
  document.querySelector(".st-header__spacer").after(box);
  let switching = null;

  async function load() {
    let sw;
    try { sw = await (await fetch("switcher", {headers: headers()})).json(); } catch (e) { return; }
    const keys = Object.keys(sw.choices || {});
    if (!sw.enabled || sw.error || keys.length < 2) { box.hidden = true; return; }
    sel.replaceChildren(...keys.map((k) => {
      const o = document.createElement("option");
      o.value = k;
      o.textContent = sw.choices[k].replace(/ \(Strata SYCL engine\)$/, "");
      o.selected = k === (sw.starting || sw.mode);
      return o;
    }));
    sel.disabled = !!sw.starting || !!switching;
    status.textContent = sw.starting && !switching ? `loading ${sw.choices[sw.starting] || sw.starting}…` : "";
    box.hidden = false;
  }

  sel.onchange = async () => {
    const mode = sel.value, name = sel.selectedOptions[0].textContent;
    if (!confirm(`Load ${name}? The current model is unloaded first; it takes about 2 minutes and stops any answer in progress.`)) {
      load();
      return;
    }
    let from = null;
    try { from = (await (await fetch("health", {cache: "no-store"})).json()).model; } catch (e) { /* going down */ }
    switching = {from, t0: Date.now()};
    sel.disabled = true;
    try {
      await fetch("switcher", {method: "POST", headers: headers(true), body: JSON.stringify({mode})});
    } catch (e) { /* this server may already be going down */ }
    const wait = async () => {
      try {
        const h = await (await fetch("health", {cache: "no-store"})).json();
        if (h.model && h.model !== switching.from) { location.reload(); return; }
      } catch (e) { /* down while the models swap */ }
      const s = Math.round((Date.now() - switching.t0) / 1000);
      if (s > 360) { status.textContent = `${name} did not come up in 6 minutes`; switching = null; sel.disabled = false; return; }
      status.textContent = `switching to ${name}… ${s} s`;
      setTimeout(wait, 3000);
    };
    setTimeout(wait, 5000);
  };

  load();
  setInterval(() => { if (!switching) load(); }, 15000);   // a switch started elsewhere shows here too
})();
