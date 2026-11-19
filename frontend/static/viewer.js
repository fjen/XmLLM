/* XmLLM static viewer — the scan with positioned overlays + inspection pane.
 *
 * Adapted from LightOnOCR (https://github.com/lighton-ai/LightOnOCR), Apache-2.0.
 * Ported from LightOnOCR's lightonocr/static/app.js rendering core:
 * percentage-positioned overlay boxes over a plain <img>, fit-to-pane zoom
 * (CSS width × factor), two-pane hover/click sync through a shared data-i,
 * IntersectionObserver .inview highlighting, and label filter chips with
 * counts. Discarded: run/job/settings management, markdown/KaTeX rendering,
 * multi-page navigation, CDN links.
 *
 * Polygons: PaddleOCR quadrilaterals carried by the projection are drawn on
 * an SVG layer (implemented, not deferred); overlays without a polygon fall
 * back to a bbox <div>. An overlay renders as one or the other, never both.
 *
 * Consumes ONLY the ViewerProjection JSON (GET /jobs/{id}/viewer) and the
 * image route (GET /jobs/{id}/image). No ALTO/PAGE XML is parsed here and no
 * business logic lives in this layer.
 */

(function () {
  'use strict';

  const SVG_NS = 'http://www.w3.org/2000/svg';

  // One colour per semantic label. XmLLM labels are categorical — block
  // roles and non-text kinds (domain/models/status.py). Line/word overlays
  // carry their OCR text as label and are coloured per level instead.
  const COLORS = {
    body: '#2563eb', heading: '#d97706', footnote: '#ea580c',
    caption: '#c026d3', margin: '#78716c', page_number: '#57534e',
    header: '#78716c', footer: '#78716c', other: '#9ca3af',
    illustration: '#7c3aed', table: '#dc2626', separator: '#57534e',
    ornament: '#db2777', graphic: '#7c3aed',
    line: '#0891b2', word: '#0d9488',
  };
  const colorOf = (label, level) =>
    COLORS[label] || (level && COLORS[level]) || '#9ca3af';

  // Overlay levels in the projection. Words are off by default: too dense at
  // page scale — the toggle turns them on.
  const LEVELS = [
    { key: 'block', title: 'Blocks' },
    { key: 'line', title: 'Lines' },
    { key: 'word', title: 'Words' },
    { key: 'non_text', title: 'Other' },
  ];

  // One open viewer at a time (the result area hosts it).
  const V = {
    jobId: null,
    data: null,          // ViewerProjection JSON
    zoom: 1,             // 1 = the whole page fits its pane
    levels: {},          // level key -> shown
    hidden: new Set(),   // labels switched off with the chips
    spy: null,           // IntersectionObserver for .inview
    root: null,          // current .xmllm-viewer element (fullscreen target)
  };

  function el(tag, cls, text) {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ---- Opening a job -----------------------------------------------------------------------------

  async function openViewer(jobId) {
    closeViewer();
    V.jobId = jobId;
    V.data = null;
    V.zoom = 1;
    V.hidden = new Set();
    V.levels = { block: true, line: true, word: false, non_text: true };

    const area = document.getElementById('viewer-area');
    if (!area) return;
    area.classList.remove('hidden');
    area.replaceChildren(el('div', 'viewer-loading', 'Loading viewer\u2026'));

    let data = null;
    try {
      const r = await fetch('/jobs/' + encodeURIComponent(jobId) + '/viewer');
      data = r.ok ? await r.json() : null;
    } catch (e) {
      data = null;
    }
    if (V.jobId !== jobId) return;  // superseded by another open while loading

    V.data = data;
    renderViewer();
  }

  function closeViewer() {
    if (V.spy) { V.spy.disconnect(); V.spy = null; }
  }

  function collectOverlays(data) {
    const out = [];
    for (const lv of LEVELS) {
      (data[lv.key + '_overlays'] || []).forEach(item => out.push({ level: lv.key, item }));
    }
    return out;
  }

  // ---- Rendering ---------------------------------------------------------------------------------

  function renderViewer() {
    // Toggles/chips rebuild the whole viewer DOM. If the old root was the
    // fullscreen element, removing it terminates fullscreen — remember and
    // re-request on the fresh root (the click is a user gesture, so the
    // request is permitted).
    const wasFullscreen = document.fullscreenElement === V.root;
    closeViewer();
    const area = document.getElementById('viewer-area');
    const data = V.data;
    area.replaceChildren();

    if (!data) {
      area.appendChild(el('div', 'viewer-nodata',
        'No viewer data available for this job (the pipeline may have failed).'));
      return;
    }

    const overlays = collectOverlays(data);
    if (!overlays.length) {
      area.appendChild(el('div', 'viewer-nodata',
        'No overlay data for this job — the scan produced no text regions.'));
      return;
    }

    const root = el('div', 'xmllm-viewer');
    V.root = root;
    root.appendChild(renderToolbar(overlays));

    const panes = el('div', 'viewer-panes');
    const left = el('div', 'viewer-pane left');
    const right = el('div', 'viewer-pane right');
    panes.appendChild(left);
    panes.appendChild(right);
    root.appendChild(panes);
    area.appendChild(root);

    // Left pane: the page image, the SVG polygon layer, the bbox boxes.
    const page = el('div', 'viewer-page');
    const img = document.createElement('img');
    img.alt = data.image_ref || 'page image';
    img.src = '/jobs/' + encodeURIComponent(V.jobId) + '/image';
    img.onerror = () => {
      const note = el('div', 'viewer-nodata',
        'Input image not available (missing, or in a format the browser cannot display).');
      page.replaceChildren(note);
    };
    page.appendChild(img);
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', 'poly');
    svg.setAttribute('viewBox', '0 0 ' + data.image_width + ' ' + data.image_height);
    svg.setAttribute('preserveAspectRatio', 'none');
    page.appendChild(svg);
    left.appendChild(page);

    // Right pane: details panel + one card per shown overlay.
    const details = el('div', 'viewer-details hidden');
    const cards = el('div', 'viewer-cards');
    right.appendChild(details);
    right.appendChild(cards);

    // Refs for refreshOverlays() — toggles/chips update these in place
    // instead of rebuilding the viewer (which would exit fullscreen).
    V.page = page;
    V.svg = svg;
    V.cards = cards;
    V.details = details;
    V.selectedId = null;

    // Hover and click sync between a box/polygon and its card (same data-i),
    // and a click opens the inspection details on the right.
    const linked = i => root.querySelectorAll('[data-i="' + CSS.escape(i) + '"]');
    root.addEventListener('mouseover', e => {
      const t = e.target.closest('[data-i]');
      if (t) linked(t.dataset.i).forEach(n => n.classList.add('on'));
    });
    root.addEventListener('mouseout', e => {
      const t = e.target.closest('[data-i]');
      if (t) linked(t.dataset.i).forEach(n => n.classList.remove('on'));
    });
    root.addEventListener('click', e => {
      const t = e.target.closest('[data-i]');
      if (!t) return;
      root.querySelectorAll('.sel').forEach(n => n.classList.remove('sel'));
      linked(t.dataset.i).forEach(n => {
        n.classList.add('sel');
        if (n !== t) n.scrollIntoView({ block: 'center', behavior: 'smooth' });
      });
      V.selectedId = t.dataset.i;
      showDetails(t.dataset.i, details);
    });

    // Boxes whose card is visible in the right pane get .inview: the page
    // shows where you are while you scroll (ported IntersectionObserver).
    V.spy = new IntersectionObserver(entries => entries.forEach(entry => {
      const box = V.page.querySelector('[data-i="' + CSS.escape(entry.target.dataset.i) + '"]');
      if (box) box.classList.toggle('inview', entry.isIntersecting);
    }), { root: right, threshold: 0.3 });

    refreshOverlays();

    fitPage();

    if (wasFullscreen && V.root.requestFullscreen) {
      V.root.requestFullscreen().catch(() => {
        // Re-request denied — stay windowed rather than failing silently.
      });
    }
  }

  function renderToolbar(overlays) {
    const bar = el('div', 'viewer-toolbar');
    const data = V.data;

    // Level toggles with counts.
    V.levelButtons = {};
    const levels = el('span', 'group');
    LEVELS.forEach(lv => {
      const count = overlays.filter(o => o.level === lv.key).length;
      const b = el('button', null, lv.title + ' ' + count);
      b.setAttribute('aria-pressed', String(!!V.levels[lv.key]));
      V.levelButtons[lv.key] = b;
      b.onclick = () => { V.levels[lv.key] = !V.levels[lv.key]; refreshOverlays(); };
      levels.appendChild(b);
    });
    bar.appendChild(levels);

    // One chip per *semantic* label, with its count; click to hide or show
    // it (ported renderFilters). Line/word overlays are labelled with their
    // OCR text — chips for those would explode the toolbar and hiding by
    // text occurrence would be meaningless — so they are level-toggled only.
    // Counts are static per document: chips stay visible even while their
    // level toggle is off (refreshOverlays keeps their pressed state true).
    const counts = {};
    overlays.forEach(o => {
      if (o.level === 'line' || o.level === 'word') return;
      const label = o.item.label || o.level;
      counts[label] = (counts[label] || 0) + 1;
    });
    const labels = Object.keys(counts);
    if (labels.length) {
      const chips = el('span', 'group');
      V.chipsEl = chips;
      labels.sort((a, b) => counts[b] - counts[a]).forEach(label => {
        const chip = el('span', 'viewer-chip');
        chip.dataset.label = label;
        chip.style.setProperty('--c', colorOf(label));
        chip.setAttribute('aria-pressed', String(!V.hidden.has(label)));
        chip.title = 'Click to hide/show';
        chip.appendChild(el('i'));
        chip.appendChild(document.createTextNode(label + ' '));
        chip.appendChild(el('b', null, String(counts[label])));
        chip.onclick = () => {
          if (V.hidden.has(label)) V.hidden.delete(label); else V.hidden.add(label);
          refreshOverlays();
        };
        chips.appendChild(chip);
      });
      bar.appendChild(chips);
    } else {
      V.chipsEl = null;
    }

    // Export-status badges (ALTO / PAGE / viewer render).
    const es = data.export_status || {};
    const badges = el('span', 'group');
    [['ALTO', es.alto_export], ['PAGE', es.page_export], ['VIEW', es.viewer_render]]
      .forEach(([name, st]) => {
        badges.appendChild(el('span', 'viewer-badge ' + (st || 'none'), name + ': ' + (st || 'none')));
      });
    bar.appendChild(badges);

    // Fit / zoom controls.
    const zoom = el('span', 'group');
    const minus = el('button', null, '\u2212');
    minus.title = 'Zoom out';
    minus.onclick = () => setZoom(V.zoom / 1.25);
    const fit = el('button', 'zoom-fit', 'Fit');
    fit.title = 'Fit the whole page in its pane';
    fit.onclick = () => setZoom(1);
    const plus = el('button', null, '+');
    plus.title = 'Zoom in';
    plus.onclick = () => setZoom(V.zoom * 1.25);
    zoom.appendChild(minus);
    zoom.appendChild(fit);
    zoom.appendChild(plus);
    bar.appendChild(zoom);

    // Fullscreen toggle (browser Fullscreen API; ESC exits). The label is
    // updated from the fullscreenchange event so it stays correct when the
    // user exits via ESC instead of the button.
    const fs = el('button', null, document.fullscreenElement
      ? '\u26f6 Exit fullscreen'
      : '\u26f6 Fullscreen');
    fs.title = 'Toggle fullscreen (ESC to exit)';
    fs.onclick = () => {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else if (V.root && V.root.requestFullscreen) {
        V.root.requestFullscreen().catch(() => {
          // Denied: permissions policy, or an iframe without allow="fullscreen".
          fs.title = 'Fullscreen unavailable in this context';
        });
      }
    };
    V.fsButton = fs;
    bar.appendChild(fs);

    // Validation flags.
    const flags = data.validation_flags || [];
    if (flags.length) {
      bar.appendChild(el('div', 'viewer-flags', 'Validation flags: ' + flags.join('; ')));
    }
    return bar;
  }

  // Re-apply level/label filters WITHOUT rebuilding the viewer: boxes,
  // polygons, cards and pressed states update in place, so the image, the
  // scroll positions and — critically — the fullscreen state stay alive.
  function refreshOverlays() {
    if (!V.data || !V.root || !V.page) return;
    const overlays = collectOverlays(V.data);

    // Level toggle pressed states.
    LEVELS.forEach(lv => {
      const b = V.levelButtons && V.levelButtons[lv.key];
      if (b) b.setAttribute('aria-pressed', String(!!V.levels[lv.key]));
    });

    // Chips: counts are static per document; only pressed state changes.
    if (V.chipsEl) {
      V.chipsEl.querySelectorAll('.viewer-chip').forEach(chip => {
        chip.setAttribute('aria-pressed', String(!V.hidden.has(chip.dataset.label)));
      });
    }

    // Boxes, polygons and cards for the currently shown set. Card pane
    // scroll is preserved across the rebuild (visible in fullscreen).
    const scroll = V.cards.scrollTop;
    V.page.querySelectorAll('.xbox').forEach(n => n.remove());
    V.svg.querySelectorAll('.poly-shape').forEach(n => n.remove());
    V.cards.replaceChildren();
    if (V.spy) V.spy.disconnect();
    renderOverlays(V.data, overlays, V.page, V.svg, V.cards);
    V.cards.scrollTop = scroll;
    V.cards.querySelectorAll('.viewer-card').forEach(c => V.spy.observe(c));

    // Re-apply the selection highlight to the rebuilt card, if still shown.
    if (V.selectedId) {
      const sel = V.cards.querySelector(
        '[data-i="' + CSS.escape(V.selectedId) + '"]');
      if (sel) sel.classList.add('sel');
    }

    // Hide the details panel if the selected overlay is no longer shown,
    // but keep V.selectedId: re-enabling the level/chip restores both the
    // .sel highlight and (on the next click) the details panel.
    if (V.selectedId && !V.details.classList.contains('hidden')) {
      const stillShown = overlays.some(o =>
        o.item.id === V.selectedId && V.levels[o.level]
        && !V.hidden.has(o.item.label || o.level));
      if (!stillShown) {
        V.details.classList.add('hidden');
      }
    }
  }

  function renderOverlays(data, overlays, page, svg, cards) {
    const W = data.image_width, H = data.image_height;
    let shown = 0;
    overlays.forEach(({ level, item }) => {
      if (!V.levels[level]) return;
      const label = item.label || level;
      if (V.hidden.has(label)) return;
      shown++;

      const c = colorOf(label, level);
      const x = item.bbox[0], y = item.bbox[1], w = item.bbox[2], h = item.bbox[3];

      if (item.polygon && item.polygon.length >= 3) {
        // Quadrilateral from the provider: SVG polygon on the overlay layer.
        const pg = document.createElementNS(SVG_NS, 'polygon');
        pg.setAttribute('class', 'poly-shape');
        pg.setAttribute('points', item.polygon.map(p => p[0] + ',' + p[1]).join(' '));
        pg.setAttribute('fill', c);
        pg.setAttribute('fill-opacity', '0.10');
        pg.setAttribute('stroke', c);
        pg.setAttribute('stroke-width', Math.max(1, W / 800));
        pg.dataset.i = item.id;
        pg.dataset.label = label;
        svg.appendChild(pg);
      } else {
        // Plain bbox: a percentage-positioned div over the image.
        const box = el('div', 'xbox');
        box.dataset.i = item.id;
        box.dataset.label = label;
        box.style.setProperty('--c', c);
        // Clamp to the page: provider quads can protrude past the image edge.
        const pct = (v, max) => Math.min(100, Math.max(0, v / max * 100));
        box.style.left = pct(x, W) + '%';
        box.style.top = pct(y, H) + '%';
        box.style.width = Math.min(100 - parseFloat(box.style.left), pct(w, W)) + '%';
        box.style.height = Math.min(100 - parseFloat(box.style.top), pct(h, H)) + '%';
        page.appendChild(box);
      }

      // Card in the inspection pane.
      const card = el('div', 'viewer-card');
      card.dataset.i = item.id;
      card.style.setProperty('--c', c);
      const meta = el('div', 'meta');
      meta.appendChild(el('span', 'lab', label));
      meta.appendChild(el('span', 'bbox', item.bbox.map(n => Math.round(n)).join(', ')));
      card.appendChild(meta);
      if (item.text) card.appendChild(el('div', 'txt', item.text));
      if (item.confidence != null) {
        card.appendChild(el('div', 'conf', (item.confidence * 100).toFixed(1) + '% confidence'));
      }
      cards.appendChild(card);
    });

    if (!shown) {
      cards.appendChild(el('div', 'viewer-empty',
        'Nothing shown — enable a level above or re-enable a label chip.'));
    }
  }

  // ---- Details (inspection_index on click) ---------------------------------------------------------

  function showDetails(id, container) {
    const d = (V.data.inspection_index || {})[id];
    container.replaceChildren();
    if (!d) {
      container.classList.add('hidden');
      return;
    }
    container.classList.remove('hidden');

    const head = el('div', 'details-head');
    head.appendChild(el('b', null, id));
    const close = el('button', null, '\u00d7');
    close.title = 'Close details';
    close.onclick = () => {
      container.replaceChildren();
      container.classList.add('hidden');
    };
    head.appendChild(close);
    container.appendChild(head);

    const rows = [
      ['Level', d.level],
      ['Text', d.text],
      ['Language', d.lang],
      ['Confidence', d.confidence != null ? (d.confidence * 100).toFixed(1) + '%' : null],
      ['Role', d.role],
      ['Readiness', d.readiness],
      ['Provenance', d.provenance_provider],
      ['Evidence', d.provenance_type],
      ['Geometry', d.geometry_status],
      ['BBox', Array.isArray(d.bbox) ? d.bbox.map(n => Math.round(n)).join(', ') : null],
      ['Polygon', Array.isArray(d.polygon) ? d.polygon.length + ' points' : null],
    ];
    const grid = el('div', 'details-grid');
    rows.forEach(([k, v]) => {
      if (v == null || v === '') return;
      grid.appendChild(el('span', 'k', k));
      grid.appendChild(el('span', 'v', String(v)));
    });
    container.appendChild(grid);

    if (d.metadata && Object.keys(d.metadata).length) {
      container.appendChild(el('pre', 'details-meta', JSON.stringify(d.metadata, null, 2)));
    }
  }

  // ---- Fit / zoom ---------------------------------------------------------------------------------

  // The page is scaled so that it fits entirely in its pane, times the zoom
  // factor. At zoom 1 the whole page is visible without scrolling; only the
  // inspection pane scrolls (ported fitPage).
  function fitPage() {
    const pane = document.querySelector('#viewer-area .viewer-pane.left');
    const page = document.querySelector('#viewer-area .viewer-page');
    const fit = document.querySelector('#viewer-area .zoom-fit');
    if (!pane || !page || !V.data) return;
    const pad = 28;  // 12px pane padding each side, plus slack
    // The pane can be display:none (hidden Scan window / failed job): its
    // client size is 0 and fitting would pin the page at the 60px floor.
    if (pane.clientWidth <= pad || pane.clientHeight <= pad) return;
    const scale = Math.min(
      (pane.clientWidth - pad) / V.data.image_width,
      (pane.clientHeight - pad) / V.data.image_height,
    );
    page.style.width = Math.max(60, Math.floor(V.data.image_width * scale * V.zoom)) + 'px';
    if (fit) {
      fit.textContent = V.zoom === 1 ? 'Fit' : V.zoom.toFixed(2).replace(/\.?0+$/, '') + '\u00d7';
    }
  }

  function setZoom(zoom) {
    V.zoom = Math.min(4, Math.max(1, zoom));
    fitPage();
  }

  window.addEventListener('resize', () => {
    if (document.querySelector('#viewer-area .viewer-page')) fitPage();
  });
  document.addEventListener('fullscreenchange', () => {
    // Label follows the actual state (ESC exits without touching the button).
    if (V.fsButton) {
      V.fsButton.textContent = document.fullscreenElement
        ? '\u26f6 Exit fullscreen'
        : '\u26f6 Fullscreen';
    }
    // Entering/exiting fullscreen changes the pane size — re-fit once the
    // new layout is applied.
    requestAnimationFrame(() => {
      if (document.querySelector('#viewer-area .viewer-page')) fitPage();
    });
  });

  // Exposed for index.html's inline script (showResult / job-list rows).
  window.openViewer = openViewer;
  window.closeViewer = closeViewer;
  // API base agreed with index.html's inline script (it sets window.API_BASE
  // right after its own `const API`); empty string when served at root.
  const API = window.API_BASE || '';
})();
