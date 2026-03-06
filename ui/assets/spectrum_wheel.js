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

  function attachWheelListener() {
    const container = document.getElementById(GRAPH_CONTAINER_ID);
    if (!container) return;

    const plotDiv = container.querySelector(".js-plotly-plot") || container;
    if (!plotDiv || plotDiv.__wheelAttached) return;

    plotDiv.__wheelAttached = true;

    plotDiv.addEventListener("wheel", (event) => {
      const deltaY = event.deltaY || 0;

      if (deltaY !== 0 && spectrumX && spectrumX.length > 1) {
        // Get current FF value from input
        const ffInput = document.getElementById(FF_INPUT_ID);
        if (!ffInput) return;

        const currentFF = parseFloat(ffInput.value) || 0;

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

        // Update FF through Dash's component update system
        if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
          window.dash_clientside.set_props(FF_INPUT_ID, { value: newFF });
        } else {
          // Fallback: update input directly and dispatch events
          ffInput.value = newFF.toFixed(3);
          ffInput.dispatchEvent(new Event('input', { bubbles: true }));
          ffInput.dispatchEvent(new Event('change', { bubbles: true }));
        }

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
        const tInput = document.getElementById(TIME_CURSOR_INPUT_ID);
        if (!tInput) return;

        const currentT = parseFloat(tInput.value) || timeX[0] || 0;

        // Move cursor center continuously in time (no sample snapping).
        const dt = Math.abs((timeX[1] || 0) - (timeX[0] || 0)) || 0.001;
        const stepSec = dt * TIME_CURSOR_WHEEL_STEP;
        const signedStep = deltaY > 0 ? -stepSec : stepSec;
        const newT = currentT + signedStep;

        if (window.dash_clientside && typeof window.dash_clientside.set_props === "function") {
          window.dash_clientside.set_props(TIME_CURSOR_INPUT_ID, { value: newT });
        } else {
          tInput.value = newT.toFixed(6);
          tInput.dispatchEvent(new Event('input', { bubbles: true }));
          tInput.dispatchEvent(new Event('change', { bubbles: true }));
        }
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
