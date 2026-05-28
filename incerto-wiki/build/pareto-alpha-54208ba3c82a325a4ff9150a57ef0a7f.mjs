function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "infinite";
  if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
    return value.toExponential(2);
  }
  return value.toFixed(2);
}

function paretoPdf(x, alpha, scale) {
  return alpha * Math.pow(scale, alpha) * Math.pow(x, -(alpha + 1));
}

function paretoSurvival(x, alpha, scale) {
  return Math.pow(scale / x, alpha);
}

function paretoQuantile(p, alpha, scale) {
  return scale * Math.pow(1 - p, -1 / alpha);
}

function svgElement(name, attributes = {}) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function pathFromPoints(points, xScale, yScale) {
  return points
    .map((point, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${xScale(point.x).toFixed(2)},${yScale(point.y).toFixed(2)}`;
    })
    .join(" ");
}

function renderPlot(svg, points, options) {
  const width = 320;
  const height = 190;
  const margin = { top: 16, right: 16, bottom: 34, left: 42 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const xScale = (x) =>
    margin.left + ((x - options.xMin) / (options.xMax - options.xMin)) * innerWidth;
  const yScale = (y) =>
    margin.top + innerHeight - ((y - options.yMin) / (options.yMax - options.yMin)) * innerHeight;

  svg.textContent = "";
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const grid = svgElement("g", { class: "grid" });
  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const y = margin.top + tick * innerHeight;
    grid.appendChild(
      svgElement("line", {
        x1: margin.left,
        x2: margin.left + innerWidth,
        y1: y,
        y2: y,
      }),
    );
  });
  svg.appendChild(grid);
  svg.appendChild(
    svgElement("line", {
      class: "axis",
      x1: margin.left,
      x2: margin.left + innerWidth,
      y1: margin.top + innerHeight,
      y2: margin.top + innerHeight,
    }),
  );
  svg.appendChild(
    svgElement("line", {
      class: "axis",
      x1: margin.left,
      x2: margin.left,
      y1: margin.top,
      y2: margin.top + innerHeight,
    }),
  );
  svg.appendChild(
    svgElement("path", {
      class: "curve",
      d: pathFromPoints(points, xScale, yScale),
    }),
  );

  const xLabel = svgElement("text", {
    class: "label",
    x: margin.left + innerWidth / 2,
    y: height - 8,
    "text-anchor": "middle",
  });
  xLabel.textContent = options.xLabel;
  svg.appendChild(xLabel);

  const yLabel = svgElement("text", {
    class: "label",
    x: 12,
    y: margin.top + innerHeight / 2,
    transform: `rotate(-90 12 ${margin.top + innerHeight / 2})`,
    "text-anchor": "middle",
  });
  yLabel.textContent = options.yLabel;
  svg.appendChild(yLabel);
}

function render({ model, el }) {
  const state = {
    alpha: Number(model.get("alpha") ?? 1.5),
    scale: Number(model.get("scale") ?? 1),
    minAlpha: Number(model.get("minAlpha") ?? 0.6),
    maxAlpha: Number(model.get("maxAlpha") ?? 5),
    step: Number(model.get("step") ?? 0.01),
  };
  state.alpha = clamp(state.alpha, state.minAlpha, state.maxAlpha);

  el.innerHTML = `
    <style>
      .widget {
        display: grid;
        gap: 1rem;
        margin: 1.5rem 0;
        padding: 1rem;
        color: #141413;
        background: #faf9f5;
        border: 1px solid #d8d4c8;
        border-radius: 8px;
        font: 16px/1.45 Charter, "Bitstream Charter", "Sitka Text", Georgia, serif;
      }
      .control {
        display: grid;
        grid-template-columns: auto minmax(0, 1fr) 4rem;
        align-items: center;
        gap: 0.75rem;
      }
      label {
        font-weight: 700;
      }
      output {
        color: #1b365d;
        font-variant-numeric: tabular-nums;
        font-weight: 700;
        text-align: right;
      }
      input[type="range"] {
        width: 100%;
        accent-color: #1b365d;
      }
      .stats {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
      }
      .stat {
        padding: 0.65rem 0;
        border-top: 1px solid #d8d4c8;
        border-bottom: 1px solid #d8d4c8;
      }
      .stat strong,
      figcaption {
        display: block;
        color: #6b6a64;
        font-size: 0.82rem;
        font-weight: 500;
        line-height: 1.2;
      }
      .stat span {
        display: block;
        margin-top: 0.2rem;
        font-size: 1.08rem;
        font-variant-numeric: tabular-nums;
      }
      .plots {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
      }
      figure {
        margin: 0;
      }
      svg {
        display: block;
        width: 100%;
        aspect-ratio: 320 / 190;
        min-height: 160px;
        background: #fffdf8;
        border: 1px solid #d8d4c8;
        border-radius: 6px;
      }
      .grid line {
        stroke: #e8e6dc;
        stroke-width: 1;
      }
      .axis {
        stroke: #6b6a64;
        stroke-width: 1.2;
      }
      .curve {
        fill: none;
        stroke: #28724f;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 2.4;
      }
      .label {
        fill: #6b6a64;
        font: 11px "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
      }
      @media (max-width: 760px) {
        .control,
        .stats,
        .plots {
          grid-template-columns: 1fr;
        }
        output {
          text-align: left;
        }
      }
    </style>
    <section class="widget">
      <div class="control">
        <label for="pareto-alpha">alpha</label>
        <input id="pareto-alpha" type="range">
        <output id="pareto-alpha-value"></output>
      </div>
      <div class="stats" aria-live="polite">
        <div class="stat"><strong>mean</strong><span id="pareto-mean"></span></div>
        <div class="stat"><strong>variance</strong><span id="pareto-variance"></span></div>
        <div class="stat"><strong>99% quantile</strong><span id="pareto-quantile"></span></div>
      </div>
      <div class="plots">
        <figure>
          <svg id="pareto-density" role="img" aria-label="Pareto density as alpha changes"></svg>
          <figcaption>Density</figcaption>
        </figure>
        <figure>
          <svg id="pareto-survival" role="img" aria-label="Pareto survival as alpha changes"></svg>
          <figcaption>Survival</figcaption>
        </figure>
        <figure>
          <svg id="pareto-quantile-plot" role="img" aria-label="Pareto quantile as alpha changes"></svg>
          <figcaption>Quantile</figcaption>
        </figure>
      </div>
    </section>
  `;

  const input = el.querySelector("#pareto-alpha");
  const alphaValue = el.querySelector("#pareto-alpha-value");
  const meanValue = el.querySelector("#pareto-mean");
  const varianceValue = el.querySelector("#pareto-variance");
  const quantileValue = el.querySelector("#pareto-quantile");
  const densitySvg = el.querySelector("#pareto-density");
  const survivalSvg = el.querySelector("#pareto-survival");
  const quantileSvg = el.querySelector("#pareto-quantile-plot");

  input.min = String(state.minAlpha);
  input.max = String(state.maxAlpha);
  input.step = String(state.step);
  input.value = String(state.alpha);

  function update(alpha) {
    const scale = state.scale;
    const xGrid = Array.from({ length: 120 }, (_, index) => scale + (19 * scale * index) / 119);
    const pGrid = Array.from({ length: 120 }, (_, index) => 0.01 + (0.98 * index) / 119);
    const density = xGrid.map((x) => ({ x, y: paretoPdf(x, alpha, scale) }));
    const survival = xGrid.map((x) => ({ x, y: paretoSurvival(x, alpha, scale) }));
    const quantile = pGrid.map((p) => ({ x: p, y: paretoQuantile(p, alpha, scale) }));
    const mean = alpha > 1 ? (alpha * scale) / (alpha - 1) : Infinity;
    const variance =
      alpha > 2 ? (alpha * scale * scale) / ((alpha - 1) ** 2 * (alpha - 2)) : Infinity;

    alphaValue.textContent = alpha.toFixed(2);
    meanValue.textContent = formatNumber(mean);
    varianceValue.textContent = formatNumber(variance);
    quantileValue.textContent = formatNumber(paretoQuantile(0.99, alpha, scale));

    renderPlot(densitySvg, density, {
      xMin: scale,
      xMax: 20 * scale,
      yMin: 0,
      yMax: Math.max(...density.map((point) => point.y)),
      xLabel: "x",
      yLabel: "f(x)",
    });
    renderPlot(survivalSvg, survival, {
      xMin: scale,
      xMax: 20 * scale,
      yMin: 0,
      yMax: 1,
      xLabel: "x",
      yLabel: "P(X > x)",
    });
    renderPlot(quantileSvg, quantile, {
      xMin: 0.01,
      xMax: 0.99,
      yMin: scale,
      yMax: Math.max(...quantile.map((point) => point.y)),
      xLabel: "p",
      yLabel: "Q(p)",
    });
  }

  const onInput = (event) => {
    const rawAlpha = Number(event.currentTarget.value);
    const alpha = clamp(rawAlpha, state.minAlpha, state.maxAlpha);
    state.alpha = alpha;
    input.value = String(alpha);
    update(alpha);

    try {
      if (typeof model.set === "function") model.set("alpha", alpha);
      if (typeof model.save_changes === "function") model.save_changes();
    } catch (error) {
      console.debug("Pareto alpha widget state was not persisted.", error);
    }
  };

  ["input", "change", "pointerup", "keyup"].forEach((eventName) => {
    input.addEventListener(eventName, onInput);
  });
  update(state.alpha);

  return () => {
    ["input", "change", "pointerup", "keyup"].forEach((eventName) => {
      input.removeEventListener(eventName, onInput);
    });
  };
}

export default { render };
