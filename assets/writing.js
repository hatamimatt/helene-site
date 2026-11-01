/* Writing cards in a magazine layout, shared by index.html (latest pieces) and articles.html (all).
   The list lives in articles/index.json; see articles/HOW-TO.md. */
const Writing = (() => {
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const fmtDate = d => new Date(d + "T12:00:00Z").toLocaleDateString("en-US", {timeZone:"UTC", year:"numeric", month:"long", day:"numeric"});

  async function load(base = "") {
    const r = await fetch(base + "articles/index.json", {cache:"no-cache"});
    if (!r.ok) throw new Error("articles/index.json");
    return (await r.json()).sort((a, b) => b.date.localeCompare(a.date));
  }
  const href = a => a.url || `article.html?slug=${encodeURIComponent(a.slug)}`;
  const external = a => /^https?:/.test(a.url || "");
  const meta = a => `<p class="w-meta">${esc(fmtDate(a.date))} · ${esc(a.minutes ? `${a.minutes} min read` : a.kind || "Article")}</p>`;
  const tags = a => (a.tags || []).length ? `<p class="w-tags">${a.tags.map(t => `<span>${esc(t)}</span>`).join("")}</p>` : "";
  const link = (a, inner, cls = "") => `<a class="${cls}" href="${esc(href(a))}"${external(a) ? ` target="_blank" rel="noopener"` : ""}>${inner}</a>`;
  const more = a => link(a, external(a) ? "Read on " + (a.kind || "the web").replace(/ article$/i, "") + " ↗" : a.url ? "Open →" : "Read article →", "w-more");

  function card(a, size) {
    const img = a.image ? link(a, `<img src="${esc(a.image)}" alt="" loading="lazy">`, "w-img") : "";
    if (size === "side") return `<article class="w-side">
        <div>${meta(a)}<h3>${link(a, esc(a.title))}</h3><p class="w-dek">${esc(a.dek)}</p></div>
        ${a.image ? link(a, `<img src="${esc(a.image)}" alt="" loading="lazy">`, "w-thumb") : ""}
      </article>`;
    return `<article class="w-${size}">${img}${meta(a)}<h3>${link(a, esc(a.title))}</h3><p class="w-dek">${esc(a.dek)}</p>${tags(a)}${more(a)}</article>`;
  }

  // lead piece large, the next one beside it, up to three more in a narrow column
  function magazine(items) {
    if (!items.length) return `<p class="w-empty">New writing is on its way.</p>`;
    const [lead, mid, ...rest] = items;
    return `<div class="w-mag${mid ? "" : " one"}${rest.length ? "" : " two"}">
        ${card(lead, "lead")}
        ${mid ? card(mid, "mid") : ""}
        ${rest.length ? `<div class="w-col">${rest.slice(0, 3).map(a => card(a, "side")).join("")}</div>` : ""}
      </div>`;
  }

  // everything after the magazine block, one row per piece
  function list(items) {
    return items.map(a => `<article class="w-row">
        <div>${meta(a)}<h3>${link(a, esc(a.title))}</h3><p class="w-dek">${esc(a.dek)}</p>${tags(a)}</div>
        ${a.image ? link(a, `<img src="${esc(a.image)}" alt="" loading="lazy">`, "w-thumb") : ""}
      </article>`).join("");
  }

  return {load, magazine, list, esc, fmtDate};
})();
