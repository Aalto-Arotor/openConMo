(function () {
  const GRAPH_CONTAINER_ID = "envelope-plot";
  const TIME_GRAPH_CONTAINER_ID = "time-plot";
  const FF_INPUT_ID = "ff_hz";
  const TIME_CURSOR_INPUT_ID = "t_cursor_s";
  const SPECTRUM_DATA_ID = "spectrum-x-hidden";
  const TIME_DATA_ID = "time-x-hidden";
  const TIME_CURSOR_WHEEL_STEP = 5;

  let spectrumX = null;
  let timeX = null;

  function setNumericInputValue(componentId, value, decimals) {
    const numericValue = Number(value);
    if (!Number.isFinite(numericValue)) return false;

    if (
      window.dash_clientside &&
      typeof window.dash_clientside.set_props === "function"
    ) {
      window.dash_clientside.set_props(componentId, { value: numericValue });
      return true;
    }

    const root = document.getElementById(componentId);
    const inputEl =
      (root && root.tagName === "INPUT" ? root : null) ||
      (root && root.querySelector ? root.querySelector("input") : null) ||
      document.querySelector(`input[id="${componentId}"]`);

    if (!inputEl) return false;

    const renderedValue =
      typeof decimals === "number"
        ? numericValue.toFixed(decimals)
        : String(numericValue);

    inputEl.value = renderedValue;
    inputEl.dispatchEvent(new Event("input", { bubbles: true }));
    inputEl.dispatchEvent(new Event("change", { bubbles: true }));
    return true;
  }

  function attachWheelListener() {
    const container = document.getElementById(GRAPH_CONTAINER_ID);
    if (!container) return;

    const plotDiv = container.querySelector(".js-plotly-plot") || container;
    if (!plotDiv || plotDiv.__wheelAttached) return;

    plotDiv.__wheelAttached = true;

    plotDiv.addEventListener("wheel", (event) => {
      const deltaY = event.deltaY || 0;

      if (deltaY !== 0 && spectrumX && spectrumX.length > 1) {
        // Get current FF value from NumberInput wrapper/input
        const ffRoot = document.getElementById(FF_INPUT_ID);
        const ffInput =
          (ffRoot && ffRoot.tagName === "INPUT" ? ffRoot : null) ||
          (ffRoot && ffRoot.querySelector ? ffRoot.querySelector("input") : null) ||
          document.querySelector(`input[id="${FF_INPUT_ID}"]`);

        const currentFF = parseFloat(ffInput && ffInput.value) || 0;

        // Find nearest bin to current FF
        let idx = 0;
        let minDist = Math.abs(spectrumX[0] - currentFF);
        for (let i = 1; i < spectrumX.length; i++) {
          const dist = Math.abs(spectrumX[i] - currentFF);
          if (dist < minDist) {
            minDist = dist;
            idx = i;
          }
        }

        // Move ±1 bin based on scroll direction (10x interpolated spectrum = very fine control)
        const step = deltaY > 0 ? -1 : 1;
        const newIdx = Math.max(0, Math.min(idx + step, spectrumX.length - 1));
        const newFF = spectrumX[newIdx];

        setNumericInputValue(FF_INPUT_ID, newFF, 3);

        console.log(`Wheel: FF ${currentFF.toFixed(3)} -> ${newFF.toFixed(3)} (bin ${idx} -> ${newIdx})`);
      }

      event.preventDefault();
      event.stopPropagation();
    }, { passive: false });
  }

  function attachTimeWheelListener() {
    const container = document.getElementById(TIME_GRAPH_CONTAINER_ID);
    if (!container) return;

    const plotDiv = container.querySelector(".js-plotly-plot") || container;
    if (!plotDiv || plotDiv.__timeWheelAttached) return;

    plotDiv.__timeWheelAttached = true;

    plotDiv.addEventListener("wheel", (event) => {
      const deltaY = event.deltaY || 0;

      // Only react when user is actively hovering this plot.
      if (!plotDiv.matches(":hover")) {
        return;
      }

      if (deltaY !== 0 && timeX && timeX.length > 1) {
        const tRoot = document.getElementById(TIME_CURSOR_INPUT_ID);
        const tInput =
          (tRoot && tRoot.tagName === "INPUT" ? tRoot : null) ||
          (tRoot && tRoot.querySelector ? tRoot.querySelector("input") : null) ||
          document.querySelector(`input[id="${TIME_CURSOR_INPUT_ID}"]`);
        if (!tInput) return;

        const currentT = parseFloat(tInput.value) || timeX[0] || 0;

        // Move cursor center continuously in time (no sample snapping).
        const dt = Math.abs((timeX[1] || 0) - (timeX[0] || 0)) || 0.001;
        const stepSec = dt * TIME_CURSOR_WHEEL_STEP;
        const signedStep = deltaY > 0 ? -stepSec : stepSec;
        const newT = currentT + signedStep;

        setNumericInputValue(TIME_CURSOR_INPUT_ID, newT, 6);
      }

      event.preventDefault();
      event.stopPropagation();
    }, { passive: false });
  }

  function loadSpectrumData() {
    const dataDiv = document.getElementById(SPECTRUM_DATA_ID);
    if (!dataDiv) return;

    const jsonText = dataDiv.textContent || dataDiv.innerText || '';
    if (jsonText) {
      try {
        spectrumX = JSON.parse(jsonText);
        console.log(`Spectrum X loaded: ${spectrumX.length} bins`);
      } catch (e) {
        console.error("Error parsing spectrum data:", e);
      }
    }
  }

  function loadTimeData() {
    const dataDiv = document.getElementById(TIME_DATA_ID);
    if (!dataDiv) return;

    const jsonText = dataDiv.textContent || dataDiv.innerText || '';
    if (jsonText) {
      try {
        timeX = JSON.parse(jsonText);
      } catch (e) {
        console.error("Error parsing time data:", e);
      }
    }
  }

  // Use MutationObserver to detect when spectrum-x-hidden updates
  function observeSpectrumData() {
    const dataDiv = document.getElementById(SPECTRUM_DATA_ID);
    if (!dataDiv) {
      setTimeout(observeSpectrumData, 100);
      return;
    }

    const observer = new MutationObserver(() => {
      loadSpectrumData();
    });

    observer.observe(dataDiv, { childList: true, characterData: true, subtree: true });

    // Load initial data
    loadSpectrumData();
  }

  function observeTimeData() {
    const dataDiv = document.getElementById(TIME_DATA_ID);
    if (!dataDiv) {
      setTimeout(observeTimeData, 100);
      return;
    }

    const observer = new MutationObserver(() => {
      loadTimeData();
    });

    observer.observe(dataDiv, { childList: true, characterData: true, subtree: true });

    // Load initial data
    loadTimeData();
  }

  // Initialize on DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      attachWheelListener();
      attachTimeWheelListener();
      observeSpectrumData();
      observeTimeData();
    });
  } else {
    attachWheelListener();
    attachTimeWheelListener();
    observeSpectrumData();
    observeTimeData();
  }

  // Retry wheel attachment every 500ms in case plot loads dynamically
  setInterval(attachWheelListener, 500);
  setInterval(attachTimeWheelListener, 500);
})();
