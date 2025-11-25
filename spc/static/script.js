// =========================================
// Helpers for percentages
// =========================================
function toPercent(val) {
    if (val == null || isNaN(val)) return "";
    return (val * 100).toFixed(1) + "%";
}

function toSignedPercent(val) {
    if (val == null || isNaN(val)) return "";
    const pct = (val * 100).toFixed(1);
    return (val > 0 ? "+" : "") + pct + "%";
}

// =========================================
// Upload CSV & initialise dashboard
// =========================================
async function uploadCSV() {
    const file = document.getElementById("fileInput").files[0];
    if (!file) return alert("Upload a CSV first!");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/run_spc", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    if (data.error) {
        console.error("Backend error:", data.error);
        alert("Error running SPC: " + data.error);
        return;
    }

    console.log("Backend response:", data);

    // Cache for filtering
    window.rawFull = data.full;
    window.rawTrend = data.trend;
    window.rawDashboard = data.dashboard;
    window.rawWales = data.wales;
    window.rawDeviations = data.deviations;

    populateFilters(window.rawFull);

    // Default Wales-wide view
    plotWalesSPC(window.rawWales);
    // renderTrendTable(window.rawTrend);
    renderDashboard(window.rawDashboard);
    renderDeviationTable(window.rawDeviations);
}

// =========================================
// Filters
// =========================================
function populateFilters(data) {
    const regionSet = new Set();
    const groupSet = new Set();

    data.forEach(d => {
        regionSet.add(d.Region);
        groupSet.add(d.group);
    });

    const regionSelect = document.getElementById("regionFilter");
    const groupSelect = document.getElementById("groupFilter");

    regionSelect.innerHTML = `<option value="ALL">All Wales</option>`;
    groupSelect.innerHTML = `<option value="ALL">All groups</option>`;

    regionSet.forEach(r => {
        regionSelect.innerHTML += `<option value="${r}">${r}</option>`;
    });

    groupSet.forEach(g => {
        groupSelect.innerHTML += `<option value="${g}">${g}</option>`;
    });
}

function applyFilters() {
    const region = document.getElementById("regionFilter").value;
    const group = document.getElementById("groupFilter").value;

    if (region === "ALL" && group === "ALL") {
        plotWalesSPC(window.rawWales);
        // renderTrendTable(window.rawTrend);
        renderDashboard(window.rawDashboard);
        // deviations table always shows all data
        return;
    }

    // Filter full dataset
    let subset = window.rawFull;
    if (region !== "ALL") subset = subset.filter(d => d.Region === region);
    if (group !== "ALL") subset = subset.filter(d => d.group === group);

    if (!subset.length) {
        alert("No data available for selected filters");
        return;
    }

    const walesSubset = buildSubsetSPC(subset);
    plotWalesSPC(walesSubset);

    const trendSubset = window.rawTrend.filter(r =>
        (region === "ALL" || r.Region === region) &&
        (group === "ALL" || r.Group === group)
    );

    const dashSubset = window.rawDashboard.filter(r =>
        (region === "ALL" || r.Region === region) &&
        (group === "ALL" || r.group === group)
    );

    // renderTrendTable(trendSubset);
    renderDashboard(dashSubset);
}

// Build mini "Wales-style" series for the filtered subset
function buildSubsetSPC(df) {
    const sorted = [...df].sort((a, b) => a.week_num - b.week_num);

    const byWeek = new Map();

    sorted.forEach(row => {
        const wk = row.week_num;
        if (!byWeek.has(wk)) {
            byWeek.set(wk, {
                week_num: wk,
                weekly_rate: row.weekly_rate,
                expected_rate: row.expected_rate,
                sigma_rate: row.sigma_rate,
                UCL: row.UCL,
                LCL: row.LCL,
                status: row.status
            });
        }
    });

    return Array.from(byWeek.values());
}

// =========================================
// Main SPC chart
// =========================================
function plotWalesSPC(walesData) {
    const canvas = document.getElementById("walesChart");
    const ctx = canvas.getContext("2d");

    if (window.walesChart && window.walesChart.destroy) {
        window.walesChart.destroy();
    }

    const weeks = walesData.map(d => d.week_num);
    const actual = walesData.map(d => d.weekly_rate);
    const expected = walesData.map(d => d.expected_rate);

    const plus1 = walesData.map(d => d.expected_rate + d.sigma_rate);
    const minus1 = walesData.map(d => d.expected_rate - d.sigma_rate);

    const plus2 = walesData.map(d => d.expected_rate + 2 * d.sigma_rate);
    const minus2 = walesData.map(d => d.expected_rate - 2 * d.sigma_rate);

    const UCL = walesData.map(d => d.UCL);
    const LCL = walesData.map(d => d.LCL);

    const colors = walesData.map(d =>
        d.status === "RED" ? "#d5281b" :
        d.status === "YELLOW" ? "#ffb81c" :
        "#007f3b"
    );

    window.walesChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: weeks,
            datasets: [
                // 2σ band
                {
                    label: "2σ bottom",
                    data: minus2,
                    borderWidth: 0,
                    pointRadius: 0
                },
                {
                    label: "±2σ band",
                    data: plus2,
                    borderWidth: 0,
                    pointRadius: 0,
                    backgroundColor: "rgba(160,160,160,0.35)",
                    fill: "-1"
                },
                // 1σ band
                {
                    label: "1σ bottom",
                    data: minus1,
                    borderWidth: 0,
                    pointRadius: 0
                },
                {
                    label: "±1σ band",
                    data: plus1,
                    borderWidth: 0,
                    pointRadius: 0,
                    backgroundColor: "rgba(200,200,200,0.35)",
                    fill: "-1"
                },
                // UCL / LCL
                {
                    label: "UCL",
                    data: UCL,
                    borderColor: "#d5281b",
                    borderDash: [6, 6],
                    borderWidth: 2,
                    pointRadius: 0
                },
                {
                    label: "LCL",
                    data: LCL,
                    borderColor: "#d5281b",
                    borderDash: [6, 6],
                    borderWidth: 2,
                    pointRadius: 0
                },
                // Expected
                {
                    label: "Expected",
                    data: expected,
                    borderColor: "#0072ce",
                    borderWidth: 3,
                    pointRadius: 0
                },
                // Actual
                {
                    label: "Actual",
                    data: actual,
                    borderColor: "#007f3b",
                    backgroundColor: colors,
                    pointRadius: 6,
                    showLine: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    labels: {
                        filter: item => !item.text.includes("bottom")
                    }
                },
                tooltip: {
                    callbacks: {
                        label: context => {
                            const label = context.dataset.label || "";
                            const val = context.parsed.y;
                            if (val == null || isNaN(val)) return label;
                            return `${label}: ${(val * 100).toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: "Week number"
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: "Weekly vaccination rate (%)"
                    },
                    ticks: {
                        callback: value => (value * 100).toFixed(0) + "%"
                    }
                }
            }
        }
    });
}

// =========================================
// Latest trends table
// =========================================
function renderTrendTable(rows) {
    const thead = `
        <tr>
            <th>Region</th>
            <th>Group</th>
            <th>Week</th>
            <th>Weekly rate</th>
            <th>Δ vs last week</th>
            <th>Δ vs ~4 weeks</th>
            <th>Status</th>
            <th>Run rule alert</th>
        </tr>`;
    document.querySelector("#trendTable thead").innerHTML = thead;

    let tbody = "";
    rows.forEach(r => {
        tbody += `
            <tr>
                <td>${r.Region}</td>
                <td>${r.Group}</td>
                <td>${r.latest_week_num}</td>
                <td>${toPercent(r.latest_weekly_rate)}</td>
                <td>${toSignedPercent(r.delta_vs_last_week)}</td>
                <td>${toSignedPercent(r.delta_vs_4weeks)}</td>
                <td>${r.latest_status}</td>
                <td>${r.latest_run_rule_alert ? "Yes" : "No"}</td>
            </tr>`;
    });

    document.querySelector("#trendTable tbody").innerHTML = tbody;
}

// =========================================
// Dashboard summary table
// =========================================
function renderDashboard(rows) {
    const thead = `
        <tr>
            <th>Region</th>
            <th>Group</th>
            <th>Total weeks</th>
            <th>Green</th>
            <th>Yellow</th>
            <th>Red</th>
            <th>Run rule alerts</th>
        </tr>`;
    document.querySelector("#dashboardTable thead").innerHTML = thead;

    // Sort by red desc, then yellow desc, then green desc
    const sorted = [...rows].sort((a, b) =>
        (b.red_weeks - a.red_weeks) ||
        (b.yellow_weeks - a.yellow_weeks) ||
        (b.green_weeks - a.green_weeks)
    );

    let tbody = "";
    sorted.forEach(r => {
        tbody += `
            <tr>
                <td>${r.Region}</td>
                <td>${r.group}</td>
                <td>${r.total_weeks}</td>
                <td>${r.green_weeks}</td>
                <td>${r.yellow_weeks}</td>
                <td>${r.red_weeks}</td>
                <td>${r.run_rule_alerts}</td>
            </tr>`;
    });

    document.querySelector("#dashboardTable tbody").innerHTML = tbody;
}

// =========================================
// Deviation table (big deviations only)
// =========================================
function renderDeviationTable(rows) {
    const thead = `
        <tr>
            <th>Region</th>
            <th>Group</th>
            <th>Week end</th>
            <th>Weekly rate</th>
            <th>Expected</th>
            <th>Sigma</th>
            <th>Classification</th>
        </tr>`;
    document.querySelector("#deviationTable thead").innerHTML = thead;

    let tbody = "";

    rows.forEach(r => {
        // Only show strong deviations / alerts
        if (
            !r.deviation_label ||
            r.deviation_label === "Within expected variation" ||
            r.deviation_label === "Insufficient data" ||
            r.deviation_label === "Slightly Lower than expected" ||
            r.deviation_label === "Slightly Higher than expected" ||
            r.deviation_label === "Higher than expected" ||
            r.deviation_label === "Lower than expected"
        ) {
            return;
        }

        const weekEnd = r.week_end
            ? new Date(r.week_end).toLocaleDateString("en-GB", {
                  day: "2-digit",
                  month: "short",
                  year: "numeric"
              })
            : "";

        tbody += `
            <tr>
                <td>${r.Region ?? r.region ?? ""}</td>
                <td>${r.group ?? r.Group ?? ""}</td>
                <td>${weekEnd}</td>
                <td>${toPercent(r.weekly_rate)}</td>
                <td>${toPercent(r.expected_rate)}</td>
                <td>${toPercent(r.sigma_rate)}</td>
                <td>${r.deviation_label}</td>
            </tr>`;
    });

    document.querySelector("#deviationTable tbody").innerHTML = tbody;
}
