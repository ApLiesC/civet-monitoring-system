const BEHAVIORS = [["active", "Active"], ["inactive", "Inactive"], ["feeding", "Feeding"]];
const daySelect = document.getElementById("day");
let EVENTS = [], DAYS = [], pieChart, barChart;

const hm = s => String(Math.floor(s / 3600) % 24).padStart(2, "0") + ":" + String(Math.floor(s % 3600 / 60)).padStart(2, "0");
const dur = s => { const m = Math.round(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`; };
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

async function load() {
  const status = document.getElementById("status");
  try {
    const res = await fetch("/api/events");
    EVENTS = await res.json();
  } catch (err) {
    status.textContent = "Could not load events. Is the API running?";
    return;
  }
  if (!EVENTS.length) status.textContent = "No events yet.";
  DAYS = [...new Set(EVENTS.map(e => e.date))].sort();
  daySelect.innerHTML = '<option value="all">All days</option>' + DAYS.map(d => `<option>${d}</option>`).join("");
  render();
}

function render() {
  const day = daySelect.value;
  const ev = EVENTS.filter(e => day === "all" || e.date === day);
  const total = b => ev.filter(e => e.behavior === b).reduce((a, e) => a + e.stop - e.start, 0);
  const count = b => ev.filter(e => e.behavior === b).length;

  document.getElementById("kpis").innerHTML = BEHAVIORS.map(([k, l]) =>
    `<div class="kpi" style="--c:var(--${k})"><b>${dur(total(k))}</b><span>${l}: ${count(k)} events</span></div>`).join("");

  const colors = BEHAVIORS.map(([k]) => css("--" + k));
  Chart.defaults.color = css("--ink");
  pieChart && pieChart.destroy();
  barChart && barChart.destroy();

  pieChart = new Chart(document.getElementById("pie"), {
    type: "doughnut",
    data: { labels: ["Active", "Inactive"],
      datasets: [{ data: [total("active"), total("inactive")], backgroundColor: [colors[0], colors[1]], borderColor: css("--panel") }] },
    options: { maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: c => `${c.label}: ${dur(c.raw)}` } } } }
  });

  barChart = new Chart(document.getElementById("bar"), {
    type: "bar",
    data: { labels: DAYS,
      datasets: BEHAVIORS.map(([k, l], i) => ({ label: l, backgroundColor: colors[i],
        data: DAYS.map(d => EVENTS.filter(e => e.date === d && e.behavior === k).length) })) },
    options: { maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, title: { display: true, text: "Times observed" }, grid: { color: css("--line") } },
                x: { grid: { display: false } } } }
  });

  document.getElementById("tables").innerHTML = BEHAVIORS.map(([k, l]) => {
    const rows = ev.filter(e => e.behavior === k).map((e, i) =>
      `<tr><td>${i + 1}</td><td>${hm(e.start)}</td><td>${hm(e.stop)}</td><td>${dur(e.stop - e.start)}</td></tr>`).join("");
    return `<section class="card"><h2><span class="tag" style="background:var(--${k})"></span>${l}</h2>
      <div class="scroll"><table><tr><th>No</th><th>Start</th><th>Stop</th><th>Duration</th></tr>${rows}</table></div></section>`;
  }).join("");
}

daySelect.addEventListener("change", render);
load();
