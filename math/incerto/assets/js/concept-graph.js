(function () {
  "use strict";

  const DATA_URL = "../data/concept-graph.json";
  const TYPE_LABELS = {
    definition: "Definitions",
    distribution: "Distributions",
    theorem: "Theorems",
    method: "Methods",
    "empirical-example": "Examples",
  };
  const EDGE_LABELS = {
    dependency: "Prerequisites",
    "proof-dependency": "Proof dependencies",
    generalization: "Special case → generalization",
    application: "Applications",
    reference: "References",
    related: "Related",
  };

  const state = {
    data: null,
    layout: "dag",
    activeTypes: new Set(),
    activeEdges: new Set(Object.keys(EDGE_LABELS)),
    selectedId: null,
    hoverId: null,
    search: "",
    depth: "all",
    positions: new Map(),
    transform: { x: 0, y: 0, k: 1 },
    dragging: null,
  };

  const svg = document.getElementById("concept-graph");
  const details = document.getElementById("graph-details");
  const searchInput = document.getElementById("graph-search");
  const layoutSelect = document.getElementById("graph-layout");
  const depthSelect = document.getElementById("graph-depth");
  const resetButton = document.getElementById("graph-reset");
  const typeFilters = document.getElementById("graph-type-filters");
  const edgeFilters = document.getElementById("graph-edge-filters");
  const svgNS = "http://www.w3.org/2000/svg";

  function createSvgElement(name, attrs) {
    const element = document.createElementNS(svgNS, name);
    Object.entries(attrs || {}).forEach(([key, value]) => {
      element.setAttribute(key, value);
    });
    return element;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function nodeById(id) {
    return state.data.nodes.find((node) => node.id === id);
  }

  function graphEdges() {
    return state.data.edges.filter((edge) => state.activeEdges.has(edge.kind));
  }

  function neighborIds(id) {
    const neighbors = new Set();
    graphEdges().forEach((edge) => {
      if (edge.source === id) neighbors.add(edge.target);
      if (edge.target === id) neighbors.add(edge.source);
    });
    return neighbors;
  }

  function localIds(rootId, depth) {
    const seen = new Set([rootId]);
    let frontier = new Set([rootId]);
    for (let step = 0; step < depth; step += 1) {
      const next = new Set();
      frontier.forEach((id) => {
        neighborIds(id).forEach((neighbor) => {
          if (!seen.has(neighbor)) {
            seen.add(neighbor);
            next.add(neighbor);
          }
        });
      });
      frontier = next;
    }
    return seen;
  }

  function matchesSearch(node) {
    if (!state.search) return true;
    const haystack = [node.title, node.id, node.type, ...(node.tags || [])]
      .join(" ")
      .toLowerCase();
    return haystack.includes(state.search);
  }

  function visibleNodeIds() {
    let ids = new Set(
      state.data.nodes
        .filter((node) => state.activeTypes.has(node.type))
        .filter(matchesSearch)
        .map((node) => node.id)
    );
    if (state.selectedId && state.depth !== "all") {
      const local = localIds(state.selectedId, Number(state.depth));
      ids = new Set([...ids].filter((id) => local.has(id)));
      if (state.activeTypes.has(nodeById(state.selectedId).type)) {
        ids.add(state.selectedId);
      }
    }
    return ids;
  }

  function visibleEdges(ids) {
    return graphEdges().filter((edge) => ids.has(edge.source) && ids.has(edge.target));
  }

  function resetTransform() {
    state.transform = { x: 0, y: 0, k: 1 };
  }

  function loadLayout() {
    const layout = state.data.layouts[state.layout];
    state.positions = new Map();
    Object.entries(layout.nodes).forEach(([id, point]) => {
      state.positions.set(id, { x: point.x, y: point.y });
    });
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    resetTransform();
  }

  function setupSvg() {
    svg.innerHTML = "";
    const defs = createSvgElement("defs");
    const marker = createSvgElement("marker", {
      id: "arrowhead",
      viewBox: "0 0 10 10",
      refX: "9",
      refY: "5",
      markerWidth: "6",
      markerHeight: "6",
      orient: "auto-start-reverse",
    });
    marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#948d80" }));
    defs.appendChild(marker);
    svg.appendChild(defs);
    svg.appendChild(createSvgElement("g", { id: "graph-viewport" }));
  }

  function focusSet() {
    const focusId = state.hoverId || state.selectedId;
    if (!focusId) return null;
    const focus = neighborIds(focusId);
    focus.add(focusId);
    return focus;
  }

  function pathForEdge(edge) {
    const source = state.positions.get(edge.source);
    const target = state.positions.get(edge.target);
    if (!source || !target) return "";
    if (state.layout === "dag" && edge.kind === "dependency") {
      const dx = Math.max(80, Math.abs(target.x - source.x) * 0.45);
      return `M ${source.x} ${source.y} C ${source.x + dx} ${source.y}, ${target.x - dx} ${target.y}, ${target.x} ${target.y}`;
    }
    return `M ${source.x} ${source.y} L ${target.x} ${target.y}`;
  }

  function renderEdges(viewport, ids, focus) {
    const group = createSvgElement("g", { class: "edges" });
    visibleEdges(ids).forEach((edge) => {
      const path = createSvgElement("path", {
        class: `edge edge-${edge.kind}`,
        d: pathForEdge(edge),
      });
      if (["dependency", "proof-dependency", "generalization", "application"].includes(edge.kind)) {
        path.setAttribute("marker-end", "url(#arrowhead)");
      }
      if (focus && focus.has(edge.source) && focus.has(edge.target)) {
        path.classList.add("edge-highlight");
      } else if (focus) {
        path.classList.add("edge-muted");
      }
      group.appendChild(path);
    });
    viewport.appendChild(group);
  }

  function shortTitle(title) {
    return title.length > 24 ? `${title.slice(0, 21)}...` : title;
  }

  function renderNodes(viewport, ids, focus) {
    const group = createSvgElement("g", { class: "nodes" });
    state.data.nodes
      .filter((node) => ids.has(node.id))
      .forEach((node) => {
        const point = state.positions.get(node.id);
        if (!point) return;
        const item = createSvgElement("g", {
          class: `node node-type-${node.type}`,
          transform: `translate(${point.x}, ${point.y})`,
          tabindex: "0",
          "data-id": node.id,
          role: "button",
          "aria-label": `Inspect ${node.title}`,
          "aria-pressed": String(node.id === state.selectedId),
        });
        if (node.id === state.selectedId) item.classList.add("node-selected");
        if (focus && focus.has(node.id)) item.classList.add("node-highlight");
        if (focus && !focus.has(node.id)) item.classList.add("node-muted");
        item.appendChild(createSvgElement("circle"));
        const layoutWidth = state.data.layouts[state.layout].width;
        const nearRightEdge = point.x > layoutWidth - 220;
        const label = createSvgElement("text", {
          x: nearRightEdge ? "-16" : "16",
          y: nearRightEdge ? "-12" : "4",
          "text-anchor": nearRightEdge ? "end" : "start",
        });
        label.textContent = shortTitle(node.title);
        item.appendChild(label);
        const title = createSvgElement("title");
        title.textContent = node.title;
        item.appendChild(title);
        item.addEventListener("pointerenter", (event) => {
          if (event.pointerType === "touch" || state.dragging) return;
          state.hoverId = node.id;
          render();
        });
        item.addEventListener("pointerleave", (event) => {
          if (event.pointerType === "touch" || state.dragging) return;
          state.hoverId = null;
          render();
        });
        item.addEventListener("click", (event) => {
          event.stopPropagation();
          state.selectedId = node.id;
          render();
        });
        item.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            event.stopPropagation();
            state.selectedId = node.id;
            render();
          }
        });
        group.appendChild(item);
      });
    viewport.appendChild(group);
  }

  function renderDetails(ids) {
    const node = state.selectedId ? nodeById(state.selectedId) : null;
    if (!node) {
      details.innerHTML = `
        <h2>Concept Graph</h2>
        <p>${state.data.stats.node_count} concepts, ${state.data.stats.dependency_edge_count} dependency edges, and ${state.data.stats.reference_edge_count} reference edges.</p>
        <p>Select a node to inspect its local neighborhood. Press Enter or Space to select a focused node, then Tab to its page link. Double-click also opens the page.</p>
        <div class="detail-meta"><span>Visible nodes</span><strong>${ids.size}</strong></div>
      `;
      return;
    }
    const dependencies = state.data.edges
      .filter((edge) => edge.kind === "dependency" && edge.target === node.id)
      .map((edge) => nodeById(edge.source).title)
      .sort();
    const usedBy = state.data.edges
      .filter((edge) => edge.kind === "dependency" && edge.source === node.id)
      .map((edge) => nodeById(edge.target).title)
      .sort();
    details.innerHTML = `
      <h2>${escapeHtml(node.title)}</h2>
      <p>${escapeHtml(TYPE_LABELS[node.type] || node.type)}</p>
      <div class="detail-meta">
        <span>Prerequisites</span><strong>${dependencies.length ? escapeHtml(dependencies.join(", ")) : "No concept dependencies"}</strong>
        <span>Used by</span><strong>${usedBy.length ? escapeHtml(usedBy.join(", ")) : "No generated dependents yet"}</strong>
      </div>
      <div class="detail-tags">
        ${(node.tags || []).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("")}
      </div>
      <p><a target="_parent" href="${escapeHtml(node.url)}">Open concept page</a></p>
    `;
  }

  function render() {
    if (!state.data) return;
    const focusedId = svg.contains(document.activeElement)
      ? document.activeElement.getAttribute("data-id") : null;
    const detailsFocused = details.contains(document.activeElement);
    const ids = visibleNodeIds();
    const focus = focusSet();
    setupSvg();
    const viewport = document.getElementById("graph-viewport");
    viewport.setAttribute(
      "transform",
      `translate(${state.transform.x} ${state.transform.y}) scale(${state.transform.k})`
    );
    renderEdges(viewport, ids, focus);
    renderNodes(viewport, ids, focus);
    renderDetails(ids);
    if (focusedId) {
      const replacement = [...svg.querySelectorAll(".node")]
        .find((node) => node.getAttribute("data-id") === focusedId);
      (replacement || searchInput).focus({ preventScroll: true });
    } else if (detailsFocused) {
      details.querySelector("a")?.focus({ preventScroll: true });
    }
  }

  function setupFilters() {
    const types = [...new Set(state.data.nodes.map((node) => node.type))].sort();
    types.forEach((type) => {
      state.activeTypes.add(type);
      const label = document.createElement("label");
      label.className = "filter-toggle";
      label.innerHTML = `<input type="checkbox" checked data-type="${escapeHtml(type)}" /> ${escapeHtml(TYPE_LABELS[type] || type)}`;
      label.querySelector("input").addEventListener("change", (event) => {
        if (event.target.checked) state.activeTypes.add(type);
        else state.activeTypes.delete(type);
        render();
      });
      typeFilters.appendChild(label);
    });

    Object.keys(EDGE_LABELS).forEach((kind) => {
      const count = state.data.edges.filter((edge) => edge.kind === kind).length;
      const label = document.createElement("label");
      label.className = "filter-toggle";
      label.innerHTML = `<input type="checkbox" checked data-edge="${kind}" /> ${EDGE_LABELS[kind]} (${count})`;
      label.querySelector("input").addEventListener("change", (event) => {
        if (event.target.checked) state.activeEdges.add(kind);
        else state.activeEdges.delete(kind);
        render();
      });
      edgeFilters.appendChild(label);
    });
  }

  function graphPointFromEvent(event) {
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const transformed = point.matrixTransform(svg.getScreenCTM().inverse());
    return {
      x: (transformed.x - state.transform.x) / state.transform.k,
      y: (transformed.y - state.transform.y) / state.transform.k,
      rawX: transformed.x,
      rawY: transformed.y,
    };
  }

  function setupInteractions() {
    searchInput.addEventListener("input", (event) => {
      state.search = event.target.value.trim().toLowerCase();
      render();
    });
    layoutSelect.addEventListener("change", (event) => {
      state.layout = event.target.value;
      loadLayout();
      render();
    });
    depthSelect.addEventListener("change", (event) => {
      state.depth = event.target.value;
      render();
    });
    resetButton.addEventListener("click", () => {
      state.selectedId = null;
      state.hoverId = null;
      searchInput.value = "";
      state.search = "";
      loadLayout();
      render();
    });
    svg.addEventListener("pointerdown", (event) => {
      if (!event.isPrimary || event.button !== 0) return;
      if (state.dragging) {
        if (event.pointerId === state.dragging.pointerId) return;
        endDrag({ type: "pointercancel", pointerId: state.dragging.pointerId });
      }
      const node = event.target.closest(".node");
      const id = node?.getAttribute("data-id");
      const point = graphPointFromEvent(event);
      state.dragging = {
        type: node ? "node" : "pan",
        id,
        pointerId: event.pointerId,
        horizontalOnly: !node && event.pointerType === "touch",
        x: point.rawX,
        y: point.rawY,
        clientX: event.clientX,
        clientY: event.clientY,
        moved: false,
        startTransform: { ...state.transform },
        startPosition: node ? { ...state.positions.get(id) } : null,
      };
      // Wait for horizontal intent on touch backgrounds so pan-y can scroll.
      if (!state.dragging.horizontalOnly) {
        // The SVG survives render(); its children are replaced on each move.
        svg.setPointerCapture(event.pointerId);
        svg.classList.add("is-dragging");
      }
    });
    svg.addEventListener("pointermove", (event) => {
      if (!state.dragging || event.pointerId !== state.dragging.pointerId) return;
      const drag = state.dragging;
      const dx = event.clientX - drag.clientX;
      const dy = event.clientY - drag.clientY;
      if (drag.horizontalOnly && !drag.moved) {
        if (Math.abs(dx) <= 4 || Math.abs(dx) <= Math.abs(dy)) return;
        svg.setPointerCapture(event.pointerId);
        svg.classList.add("is-dragging");
      }
      const point = graphPointFromEvent(event);
      if (Math.hypot(dx, dy) > 4) drag.moved = true;
      if (drag.type === "pan") {
        state.transform.x = drag.startTransform.x + (point.rawX - drag.x);
        state.transform.y = drag.startTransform.y + (drag.horizontalOnly ? 0 : point.rawY - drag.y);
      } else if (drag.type === "node") {
        state.positions.set(drag.id, { x: point.x, y: point.y });
      }
      render();
    });
    function endDrag(event) {
      if (!state.dragging || event.pointerId !== state.dragging.pointerId) return;
      const drag = state.dragging;
      state.dragging = null;
      svg.classList.remove("is-dragging");
      if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
      if (event.type !== "pointerup") {
        state.transform = drag.startTransform;
        if (drag.type === "node") state.positions.set(drag.id, drag.startPosition);
        render();
        return;
      }
      // Capture can retarget the click to the SVG, so handle node taps here.
      if (drag.type === "node" && !drag.moved) {
        state.selectedId = drag.id;
        render();
      }
    }
    // Uncaptured touch gestures may finish outside the SVG.
    window.addEventListener("pointerup", endDrag, true);
    window.addEventListener("pointercancel", endDrag, true);
    svg.addEventListener("pointerleave", (event) => {
      if (!svg.hasPointerCapture(event.pointerId)) endDrag(event);
    });
    svg.addEventListener("lostpointercapture", endDrag);
    svg.addEventListener("dblclick", (event) => {
      const node = document.elementFromPoint(event.clientX, event.clientY)?.closest(".node");
      if (node && svg.contains(node)) window.open(nodeById(node.getAttribute("data-id")).url, "_parent");
    });
    svg.addEventListener("wheel", (event) => {
      event.preventDefault();
      const before = graphPointFromEvent(event);
      const factor = event.deltaY > 0 ? 0.9 : 1.1;
      const nextScale = Math.min(3.2, Math.max(0.35, state.transform.k * factor));
      state.transform.k = nextScale;
      state.transform.x = before.rawX - before.x * nextScale;
      state.transform.y = before.rawY - before.y * nextScale;
      render();
    }, { passive: false });
  }

  function showError(error) {
    details.innerHTML = `<h2>Graph unavailable</h2><p>${escapeHtml(error.message || error)}</p>`;
  }

  fetch(DATA_URL)
    .then((response) => {
      if (!response.ok) throw new Error(`Could not load ${DATA_URL}`);
      return response.json();
    })
    .then((data) => {
      state.data = data;
      setupFilters();
      setupInteractions();
      loadLayout();
      render();
    })
    .catch(showError);
})();
