const BEHAVIORS = [["active", "Active"], ["inactive", "Inactive"], ["feeding", "Feeding"]];
const weekSelect = document.getElementById("week");
const daySelect = document.getElementById("day");
let EVENTS = [], WEEKS = [], pieChart, barChart;

const hm = s => String(Math.floor(s / 3600) % 24).padStart(2, "0") + ":" + String(Math.floor(s % 3600 / 60)).padStart(2, "0");
const dur = s => { const m = Math.round(s / 60); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m} min`; };
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const pad = n => String(n).padStart(2, "0");
const iso = d => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
const parse = s => { const [y, m, d] = s.split("-").map(Number); return new Date(y, m - 1, d); };
const addDays = (d, n) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
const mondayOf = d => addDays(d, -((d.getDay() + 6) % 7));
const fmt = (d, o) => d.toLocaleDateString("en-GB", o);
const weekLabel = start => {
  const end = addDays(start, 6);
  return `${fmt(start, { day: "numeric", month: "short" })} – ${fmt(end, { day: "numeric", month: "short", year: "numeric" })}`;
};
const weekDays = start => Array.from({ length: 7 }, (_, i) => iso(addDays(start, i)));

async function load() {
  const status = document.getElementById("status");
  try {
    EVENTS = await (await fetch("/api/events")).json();
  } catch (err) {
    status.textContent = "Could not load events. Is the API running?";
    return;
  }
  if (!EVENTS.length) { status.textContent = "No events yet."; }

  const dates = EVENTS.map(e => e.date).sort();
  const first = mondayOf(dates.length ? parse(dates[0]) : new Date());
  const last = mondayOf(dates.length ? parse(dates[dates.length - 1]) : new Date());
  WEEKS = [];
  for (let w = first; w <= last; w = addDays(w, 7)) WEEKS.push(iso(w));

  weekSelect.innerHTML = WEEKS.map(w => `<option value="${w}">${weekLabel(parse(w))}</option>`).join("");
  weekSelect.value = WEEKS[WEEKS.length - 1]; // newest week
  setWeek();
}

function setWeek() {
  const days = weekDays(parse(weekSelect.value));
  daySelect.innerHTML = '<option value="all">Whole week</option>' +
    days.map(d => `<option value="${d}">${fmt(parse(d), { weekday: "short", day: "numeric", month: "short" })}</option>`).join("");
  render();
}

function shiftWeek(n) {
  const i = WEEKS.indexOf(weekSelect.value) + n;
  if (i < 0 || i >= WEEKS.length) return;
  weekSelect.value = WEEKS[i];
  setWeek();
}

function render() {
  const days = weekDays(parse(weekSelect.value));
  const weekEvents = EVENTS.filter(e => days.includes(e.date));
  const day = daySelect.value;
  const ev = weekEvents.filter(e => day === "all" || e.date === day);
  const total = b => ev.filter(e => e.behavior === b).reduce((a, e) => a + e.stop - e.start, 0);
  const count = b => ev.filter(e => e.behavior === b).length;

  const i = WEEKS.indexOf(weekSelect.value);
  document.getElementById("prev").disabled = i <= 0;
  document.getElementById("next").disabled = i >= WEEKS.length - 1;

  document.getElementById("kpis").innerHTML = BEHAVIORS.map(([k, l]) =>
    `<div class="kpi" style="--c:var(--${k})"><b>${dur(total(k))}</b><span>${l}: ${count(k)} events</span></div>`).join("");

  const colors = BEHAVIORS.map(([k]) => css("--" + k));
  Chart.defaults.color = css("--ink");
  pieChart && pieChart.destroy();
  barChart && barChart.destroy();

  const noData = total("active") + total("inactive") === 0;
  pieChart = new Chart(document.getElementById("pie"), {
    type: "doughnut",
    data: { labels: noData ? ["No data"] : ["Active", "Inactive"],
      datasets: [{ data: noData ? [1] : [total("active"), total("inactive")],
        backgroundColor: noData ? [css("--line")] : [colors[0], colors[1]], borderColor: css("--panel") }] },
    options: { maintainAspectRatio: false, plugins: { tooltip: { enabled: !noData, callbacks: { label: c => `${c.label}: ${dur(c.raw)}` } } } }
  });

  // if a day has no data then show 0
  const labels = days.map(d => fmt(parse(d), { weekday: "short", day: "numeric", month: "short" }));
  barChart = new Chart(document.getElementById("bar"), {
    type: "bar",
    data: { labels,
      datasets: BEHAVIORS.map(([k, l], j) => ({ label: l, backgroundColor: colors[j],
        data: days.map(d => weekEvents.filter(e => e.date === d && e.behavior === k).length) })) },
    options: { maintainAspectRatio: false,
      scales: { y: { beginAtZero: true, ticks: { precision: 0 }, title: { display: true, text: "Times observed" }, grid: { color: css("--line") } },
                x: { grid: { display: false } } } }
  });

  document.getElementById("tables").innerHTML = BEHAVIORS.map(([k, l]) => {
    const evs = ev.filter(e => e.behavior === k);
    const rows = evs.length
      ? evs.map((e, n) => `<tr><td>${n + 1}</td><td>${hm(e.start)}</td><td>${hm(e.stop)}</td><td>${dur(e.stop - e.start)}</td></tr>`).join("")
      : `<tr><td colspan="4">No data</td></tr>`;
    return `<section class="card"><h2><span class="tag" style="background:var(--${k})"></span>${l}</h2>
      <div class="scroll"><table><tr><th>No</th><th>Start</th><th>Stop</th><th>Duration</th></tr>${rows}</table></div></section>`;
  }).join("");
}

weekSelect.addEventListener("change", setWeek);
daySelect.addEventListener("change", render);
document.getElementById("prev").addEventListener("click", () => shiftWeek(-1));
document.getElementById("next").addEventListener("click", () => shiftWeek(1));
load();