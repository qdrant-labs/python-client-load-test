import json
from datetime import datetime
from typing import List, Optional

import numpy as np

from .metrics import compute_metric, recall_histogram_data
from .models import ExperimentRun, RunResult, SweepConfig

_SHARED_HEAD = """
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f1117; color: #e2e8f0; }
  .container { max-width: 1200px; margin: 0 auto; padding: 32px 24px; }
  h1 { font-size: 1.75rem; font-weight: 700; color: #f8fafc; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 32px; }
  h2 { font-size: 1.05rem; font-weight: 600; color: #cbd5e1; margin-bottom: 16px; letter-spacing: 0.05em; text-transform: uppercase; }
  .section { margin-bottom: 36px; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 36px; }
  .card { background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 20px 24px; }
  .card-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
  .card-value { font-size: 2rem; font-weight: 700; color: #f1f5f9; line-height: 1; }
  .card-sub { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; }
  .card.highlight .card-value { color: #38bdf8; }
  .card.good .card-value { color: #4ade80; }
  .card.warn .card-value { color: #fb923c; }
  .chart-box { background: #1e2433; border: 1px solid #2d3748; border-radius: 12px; padding: 24px; margin-bottom: 24px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
  thead th { background: #0f1117; color: #64748b; font-weight: 600; text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.06em; padding: 10px 12px; text-align: left; border-bottom: 1px solid #2d3748; }
  tbody td { padding: 9px 12px; border-bottom: 1px solid #1a2030; color: #cbd5e1; }
  tbody tr:hover td { background: #263044; }
  .config-table td:first-child { color: #64748b; font-weight: 500; width: 220px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; }
  .badge-blue { background: #1e3a5f; color: #38bdf8; }
  .badge-green { background: #14532d; color: #4ade80; }
  .badge-amber { background: #422006; color: #fb923c; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 700px) { .two-col { grid-template-columns: 1fr; } }
  /* Tabs */
  .tab-bar { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 16px; }
  .tab-btn { background: #1e2433; border: 1px solid #2d3748; border-radius: 8px; color: #94a3b8;
    padding: 6px 14px; font-size: 0.8rem; cursor: pointer; transition: background 0.15s; }
  .tab-btn:hover { background: #263044; }
  .tab-btn.active { background: #263044; border-color: #38bdf8; color: #38bdf8; font-weight: 600; }
  .hist-panel { display: none; }
  .hist-panel.active { display: block; }
  /* Stopped-run highlight */
  tr.stopped-run td { border-left: 3px solid #4ade80; }
  tr.stopped-run td:first-child { padding-left: 9px; }
</style>
"""

_CHART_DEFAULTS_JS = """
const chartDefaults = {
  responsive: true,
  plugins: {
    legend: { labels: { color: '#94a3b8', font: { size: 12 } } },
    tooltip: { backgroundColor: '#1e2433', titleColor: '#f1f5f9', bodyColor: '#cbd5e1',
                borderColor: '#2d3748', borderWidth: 1 },
  },
  scales: {
    x: { ticks: { color: '#64748b' }, grid: { color: '#1e2433' } },
    y: { ticks: { color: '#64748b' }, grid: { color: '#2d3748' } },
  },
};

function makeHistChart(canvasId, run) {
  const histLabels = Array.from({length: 100}, (_, i) => `${i}%`);
  new Chart(document.getElementById(canvasId), {
    type: 'bar',
    data: {
      labels: histLabels,
      datasets: [{
        label: 'Query count',
        data: run.counts,
        backgroundColor: histLabels.map((_, i) => {
          if (i <= run.p01Bucket) return 'rgba(239,68,68,0.75)';
          if (i <= run.p05Bucket) return 'rgba(249,115,22,0.75)';
          if (i <= run.p10Bucket) return 'rgba(234,179,8,0.75)';
          if (i <= run.p50Bucket) return 'rgba(56,189,248,0.75)';
          return 'rgba(74,222,128,0.65)';
        }),
        borderColor: histLabels.map((_, i) => {
          if (i <= run.p01Bucket) return '#ef4444';
          if (i <= run.p05Bucket) return '#f97316';
          if (i <= run.p10Bucket) return '#eab308';
          if (i <= run.p50Bucket) return '#38bdf8';
          return '#4ade80';
        }),
        borderWidth: 1, borderRadius: 2,
        categoryPercentage: 1.0, barPercentage: 1.0,
      }],
    },
    options: {
      ...chartDefaults,
      plugins: {
        ...chartDefaults.plugins,
        legend: { display: false },
        annotation: {
          annotations: {
            lineP01: {
              type: 'line', xMin: run.p01Bucket, xMax: run.p01Bucket,
              borderColor: '#ef4444', borderWidth: 2, borderDash: [5,3],
              label: { content: 'p1: ' + (run.p01Recall*100).toFixed(1)+'%', display: true,
                position: 'start', backgroundColor: 'rgba(239,68,68,0.15)', color: '#ef4444',
                font: {size:11,weight:'bold'}, padding: {x:6,y:3}, yAdjust: 0 },
            },
            lineP05: {
              type: 'line', xMin: run.p05Bucket, xMax: run.p05Bucket,
              borderColor: '#f97316', borderWidth: 2, borderDash: [5,3],
              label: { content: 'p5: ' + (run.p05Recall*100).toFixed(1)+'%', display: true,
                position: 'start', backgroundColor: 'rgba(249,115,22,0.15)', color: '#f97316',
                font: {size:11,weight:'bold'}, padding: {x:6,y:3}, yAdjust: 28 },
            },
            lineP10: {
              type: 'line', xMin: run.p10Bucket, xMax: run.p10Bucket,
              borderColor: '#eab308', borderWidth: 2, borderDash: [5,3],
              label: { content: 'p10: ' + (run.p10Recall*100).toFixed(1)+'%', display: true,
                position: 'start', backgroundColor: 'rgba(234,179,8,0.15)', color: '#eab308',
                font: {size:11,weight:'bold'}, padding: {x:6,y:3}, yAdjust: 56 },
            },
            lineP50: {
              type: 'line', xMin: run.p50Bucket, xMax: run.p50Bucket,
              borderColor: '#38bdf8', borderWidth: 2, borderDash: [5,3],
              label: { content: 'p50: ' + (run.p50Recall*100).toFixed(1)+'%', display: true,
                position: 'start', backgroundColor: 'rgba(56,189,248,0.15)', color: '#38bdf8',
                font: {size:11,weight:'bold'}, padding: {x:6,y:3}, yAdjust: 84 },
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: '#64748b',
            callback: (v, i) => i % 10 === 0 ? histLabels[i] : '',
            maxRotation: 0 },
          grid: { color: '#1e2433' },
          title: { display: true, text: 'Recall', color: '#64748b' },
        },
        y: { ticks: { color: '#64748b' }, grid: { color: '#2d3748' },
          title: { display: true, text: 'Queries', color: '#64748b' } },
      },
    },
  });
}
"""


def _quant_label(run: ExperimentRun) -> str:
    if not run.collection_config or not run.collection_config.quantization:
        return "—"
    cc = run.collection_config
    q = cc.quantization
    if q == "turbo":
        return f"TurboQuant {cc.bits or 'bits4'}"
    if q == "binary":
        enc = {"two_bits": "2-bit", "one_and_half_bits": "1.5-bit"}.get(cc.encoding or "", "1-bit")
        qenc = f" / {cc.query_encoding}" if cc.query_encoding and cc.query_encoding not in ("default", "binary") else ""
        return f"binary {enc}{qenc}"
    return q


def _search_label(run: ExperimentRun) -> str:
    parts = []
    if getattr(run, "query_mode", "dense") != "dense":
        parts.append(run.query_mode)
    if run.hnsw_ef is not None:
        parts.append(f"ef={run.hnsw_ef}")
    if run.oversampling is not None:
        parts.append(f"os={run.oversampling}x")
    if run.rescore:
        parts.append("rescore")
    if getattr(run, "query_filter", None):
        parts.append("filtered")
    return ", ".join(parts) or "defaults"


def generate_html_report(
    qdrant_url: str,
    collection_name: str,
    vector_name: Optional[str],
    num_queries: int,
    num_batches: int,
    concurrency: int,
    limit: int,
    rescore: bool,
    prefer_grpc: bool,
    output_path: str,
    ann_qps: List[float],
    exact_qps: List[float],
    ann_time: float,
    exact_time: float,
    ann_latencies: List[float],
    exact_latencies: List[float],
    recalls: List[float],
    query_source: str,
    num_embeddings,
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mean_ann_qps = float(np.mean(ann_qps)) if ann_qps else 0
    mean_exact_qps = float(np.mean(exact_qps)) if exact_qps else 0
    mean_recall = float(np.mean(recalls)) if recalls else 0
    p01_recall = float(np.percentile(recalls, 1)) if recalls else 0
    p05_recall = float(np.percentile(recalls, 5)) if recalls else 0
    p10_recall = float(np.percentile(recalls, 10)) if recalls else 0
    p50_recall = float(np.percentile(recalls, 50)) if recalls else 0
    p95_recall = float(np.percentile(recalls, 95)) if recalls else 0
    p99_recall = float(np.percentile(recalls, 99)) if recalls else 0

    p50_ann_lat = float(np.percentile(ann_latencies, 50)) * 1000 if ann_latencies else 0
    p95_ann_lat = float(np.percentile(ann_latencies, 95)) * 1000 if ann_latencies else 0
    p50_exact_lat = float(np.percentile(exact_latencies, 50)) * 1000 if exact_latencies else 0
    p95_exact_lat = float(np.percentile(exact_latencies, 95)) * 1000 if exact_latencies else 0

    hist_json = json.dumps(recall_histogram_data(recalls))
    batch_labels = json.dumps([str(i + 1) for i in range(max(len(ann_qps), len(exact_qps)))])
    ann_qps_json = json.dumps([round(q, 2) for q in ann_qps] + [0] * max(0, len(exact_qps) - len(ann_qps)))
    exact_qps_json = json.dumps([round(q, 2) for q in exact_qps] + [0] * max(0, len(ann_qps) - len(exact_qps)))

    url_display = qdrant_url[:60] + ("..." if len(qdrant_url) > 60 else "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>Qdrant Load Test Report</title>
{_SHARED_HEAD}
</head>
<body>
<div class="container">

  <h1>Qdrant Load Test Report</h1>
  <div class="subtitle">Generated: {ts} &nbsp;|&nbsp; Collection: <strong>{collection_name}</strong></div>

  <div class="cards">
    <div class="card highlight">
      <div class="card-label">ANN QPS (avg)</div>
      <div class="card-value">{mean_ann_qps:.0f}</div>
      <div class="card-sub">Total time: {ann_time:.2f}s</div>
    </div>
    <div class="card highlight">
      <div class="card-label">Exact QPS (avg)</div>
      <div class="card-value">{mean_exact_qps:.0f}</div>
      <div class="card-sub">Total time: {exact_time:.2f}s</div>
    </div>
    <div class="card {'good' if mean_recall >= 0.9 else 'warn'}">
      <div class="card-label">Mean Recall@{limit}</div>
      <div class="card-value">{mean_recall*100:.1f}%</div>
      <div class="card-sub">p1: {p01_recall*100:.1f}% &nbsp;|&nbsp; p10: {p10_recall*100:.1f}% &nbsp;|&nbsp; p50: {p50_recall*100:.1f}%</div>
    </div>
    <div class="card">
      <div class="card-label">Total Queries</div>
      <div class="card-value">{num_queries}</div>
      <div class="card-sub">Limit: {limit} &nbsp;|&nbsp; Batches: {num_batches}</div>
    </div>
    <div class="card">
      <div class="card-label">Concurrency</div>
      <div class="card-value">{concurrency}</div>
      <div class="card-sub">Vector: {vector_name or "default"}</div>
    </div>
  </div>

  <div class="section">
    <h2>Configuration</h2>
    <div class="chart-box">
      <table class="config-table">
        <tbody>
          <tr><td>Qdrant URL</td><td>{url_display}</td></tr>
          <tr><td>Collection</td><td>{collection_name}</td></tr>
          <tr><td>Vector Name</td><td>{vector_name or '<em>default</em>'}</td></tr>
          <tr><td>Query Source</td><td>{query_source}</td></tr>
          <tr><td>Embeddings Available</td><td>{num_embeddings}</td></tr>
          <tr><td>Total Queries</td><td>{num_queries}</td></tr>
          <tr><td>Num Batches</td><td>{num_batches}</td></tr>
          <tr><td>Concurrency</td><td>{concurrency}</td></tr>
          <tr><td>Limit (top-k)</td><td>{limit}</td></tr>
          <tr><td>Rescore</td><td>{rescore}</td></tr>
          <tr><td>Transport</td><td>{'gRPC' if prefer_grpc else 'REST'}</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>QPS per Batch</h2>
    <div class="chart-box">
      <canvas id="qpsChart" height="100"></canvas>
    </div>
  </div>

  <div class="two-col section">
    <div>
      <h2>Latency (ms)</h2>
      <div class="chart-box">
        <table>
          <thead><tr><th>Phase</th><th>p50</th><th>p95</th></tr></thead>
          <tbody>
            <tr><td><span class="badge badge-blue">ANN</span></td><td>{p50_ann_lat:.1f} ms</td><td>{p95_ann_lat:.1f} ms</td></tr>
            <tr><td><span class="badge badge-green">Exact</span></td><td>{p50_exact_lat:.1f} ms</td><td>{p95_exact_lat:.1f} ms</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <div>
      <h2>Recall@{limit} Stats</h2>
      <div class="chart-box">
        <table>
          <thead><tr><th>Metric</th><th>Value</th></tr></thead>
          <tbody>
            <tr style="background:rgba(239,68,68,0.10)"><td style="color:#ef4444;font-weight:700">p1 <span style="font-size:0.7rem;color:#94a3b8;font-weight:400">(worst 1%)</span></td><td style="color:#ef4444;font-weight:700">{p01_recall*100:.2f}%</td></tr>
            <tr style="background:rgba(249,115,22,0.10)"><td style="color:#f97316;font-weight:700">p5</td><td style="color:#f97316;font-weight:700">{p05_recall*100:.2f}%</td></tr>
            <tr style="background:rgba(234,179,8,0.10)"><td style="color:#eab308;font-weight:700">p10</td><td style="color:#eab308;font-weight:700">{p10_recall*100:.2f}%</td></tr>
            <tr style="background:rgba(56,189,248,0.10)"><td style="color:#38bdf8;font-weight:700">p50 <span style="font-size:0.7rem;color:#94a3b8;font-weight:400">(median)</span></td><td style="color:#38bdf8;font-weight:700">{p50_recall*100:.2f}%</td></tr>
            <tr><td>Mean</td><td>{mean_recall*100:.2f}%</td></tr>
            <tr><td>p99</td><td>{p99_recall*100:.2f}%</td></tr>
            <tr><td>Queries scored</td><td>{len(recalls)}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Recall@{limit} Distribution</h2>
    <div class="chart-box">
      <canvas id="recallChart" height="100"></canvas>
    </div>
  </div>

  <div class="section">
    <h2>Per-Batch Results</h2>
    <div class="chart-box">
      <table>
        <thead><tr><th>Batch</th><th>ANN QPS</th><th>Exact QPS</th><th>QPS Ratio (ANN/Exact)</th></tr></thead>
        <tbody>
          {''.join(
              f'<tr><td>{i+1}</td><td>{aq:.2f}</td><td>{eq:.2f}</td><td>{(aq/eq if eq > 0 else 0):.2f}x</td></tr>'
              for i, (aq, eq) in enumerate(zip(
                  [round(q,2) for q in ann_qps] + [0]*max(0,len(exact_qps)-len(ann_qps)),
                  [round(q,2) for q in exact_qps] + [0]*max(0,len(ann_qps)-len(exact_qps)),
              ))
          )}
        </tbody>
      </table>
    </div>
  </div>

</div>
<script>
{_CHART_DEFAULTS_JS}

new Chart(document.getElementById('qpsChart'), {{
  type: 'bar',
  data: {{
    labels: {batch_labels},
    datasets: [
      {{ label: 'ANN QPS', data: {ann_qps_json}, backgroundColor: 'rgba(56,189,248,0.7)', borderColor: '#38bdf8', borderWidth: 1 }},
      {{ label: 'Exact QPS', data: {exact_qps_json}, backgroundColor: 'rgba(74,222,128,0.7)', borderColor: '#4ade80', borderWidth: 1 }},
    ],
  }},
  options: {{ ...chartDefaults }},
}});

makeHistChart('recallChart', {hist_json});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"\n✅ HTML report saved to: {output_path}")


def generate_sweep_html_report(
    results: List[RunResult],
    sweep: SweepConfig,
    stopping_run_index: Optional[int],
    collection_name: str,
    num_queries: int,
    limit: int,
    concurrency: int,
    output_path: str,
):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_planned = len(sweep.runs)
    num_executed = len(results)
    stopped_early = stopping_run_index is not None
    final_stop_val = compute_metric(results[-1].recalls, sweep.stopping_metric) if results else 0.0

    run_names_json = json.dumps([r.run.name for r in results])
    p01_json = json.dumps([round(r.p01 * 100, 2) for r in results])
    p05_json = json.dumps([round(r.p05 * 100, 2) for r in results])
    p10_json = json.dumps([round(r.p10 * 100, 2) for r in results])
    p50_json = json.dumps([round(r.p50 * 100, 2) for r in results])
    qps_json = json.dumps([round(r.qps, 1) for r in results])
    hist_data_json = json.dumps([
        {**recall_histogram_data(r.recalls), "name": r.run.name}
        for r in results
    ])

    stop_run_idx = stopping_run_index if stopping_run_index is not None else -1

    def _pct_cell(v: float, color: str) -> str:
        return f'<td style="color:{color};font-weight:700">{v*100:.2f}%</td>'

    table_rows = ""
    for i, r in enumerate(results):
        stopped = (
            stopped_early
            and stopping_run_index is not None
            and r.run.name == results[stopping_run_index].run.name
        )
        row_class = ' class="stopped-run"' if stopped else ""
        stop_badge = ' <span class="badge badge-green">✓ stop</span>' if stopped else ""
        table_rows += f"""<tr{row_class}>
          <td>{i+1}{stop_badge}</td>
          <td style="font-weight:600;color:#f1f5f9">{r.run.name}</td>
          <td>{_quant_label(r.run)}</td>
          <td>{r.run.hnsw_ef or '—'}</td>
          <td>{r.run.oversampling or '—'}</td>
          <td>{'✓' if r.run.rescore else '—'}</td>
          <td style="color:#38bdf8">{r.qps:.0f}</td>
          {_pct_cell(r.p01, '#ef4444')}
          {_pct_cell(r.p05, '#f97316')}
          {_pct_cell(r.p10, '#eab308')}
          {_pct_cell(r.p50, '#38bdf8')}
          <td>{r.mean_recall*100:.2f}%</td>
        </tr>"""

    tab_buttons = "".join(
        f'<button class="tab-btn{" active" if i==0 else ""}" onclick="showHistTab({i})">{i+1}: {r.run.name}</button>'
        for i, r in enumerate(results)
    )
    hist_panels = "".join(
        f'<div id="hist-panel-{i}" class="hist-panel chart-box{" active" if i==0 else ""}"><canvas id="hist-{i}" height="100"></canvas></div>'
        for i in range(len(results))
    )

    stop_annotation_js = ""
    if stopped_early and stop_run_idx >= 0:
        stop_annotation_js = f"""
        annotation: {{
          annotations: {{
            stopLine: {{
              type: 'line', xMin: {stop_run_idx}, xMax: {stop_run_idx},
              borderColor: '#4ade80', borderWidth: 2, borderDash: [6,3],
              label: {{
                content: 'Stop ({sweep.stopping_metric} ≥ {sweep.stopping_threshold*100:.0f}%)',
                display: true, position: 'start', backgroundColor: 'rgba(74,222,128,0.15)',
                color: '#4ade80', font: {{size:11,weight:'bold'}}, padding: {{x:6,y:3}},
              }},
            }},
          }},
        }},"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<title>Qdrant Recall Sweep Report</title>
{_SHARED_HEAD}
</head>
<body>
<div class="container">

  <h1>Qdrant Recall Sweep Report</h1>
  <div class="subtitle">
    Generated: {ts} &nbsp;|&nbsp;
    Collection: <strong>{collection_name}</strong> &nbsp;|&nbsp;
    Stop: <strong>{sweep.stopping_metric} ≥ {(sweep.stopping_threshold or 0)*100:.1f}%</strong>
  </div>

  <div class="cards">
    <div class="card highlight">
      <div class="card-label">Runs Executed</div>
      <div class="card-value">{num_executed}</div>
      <div class="card-sub">of {num_planned} planned</div>
    </div>
    <div class="card {'good' if stopped_early else 'warn'}">
      <div class="card-label">Stopping Condition</div>
      <div class="card-value">{'Met ✓' if stopped_early else 'Not met'}</div>
      <div class="card-sub">{sweep.stopping_metric} = {final_stop_val*100:.2f}% at final run</div>
    </div>
    <div class="card">
      <div class="card-label">Queries / Run</div>
      <div class="card-value">{num_queries}</div>
      <div class="card-sub">Limit: {limit} &nbsp;|&nbsp; Concurrency: {concurrency}</div>
    </div>
    <div class="card">
      <div class="card-label">Best p10 Recall</div>
      <div class="card-value">{max((r.p10 for r in results), default=0)*100:.1f}%</div>
      <div class="card-sub">Across {num_executed} run(s)</div>
    </div>
    <div class="card">
      <div class="card-label">Best QPS</div>
      <div class="card-value">{max((r.qps for r in results), default=0):.0f}</div>
      <div class="card-sub">Peak ANN throughput</div>
    </div>
  </div>

  <!-- Comparison table -->
  <div class="section">
    <h2>Run Comparison</h2>
    <div class="chart-box" style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Run</th>
            <th>Quantization</th>
            <th>hnsw_ef</th>
            <th>Oversample</th>
            <th>Rescore</th>
            <th>QPS</th>
            <th style="color:#ef4444">p1%</th>
            <th style="color:#f97316">p5%</th>
            <th style="color:#eab308">p10%</th>
            <th style="color:#38bdf8">p50%</th>
            <th>Mean%</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
  </div>

  <!-- Recall trend chart -->
  <div class="section">
    <h2>Recall Trend across Runs</h2>
    <div class="chart-box">
      <canvas id="trendChart" height="80"></canvas>
    </div>
  </div>

  <!-- QPS comparison -->
  <div class="section">
    <h2>QPS per Run</h2>
    <div class="chart-box">
      <canvas id="qpsChart" height="80"></canvas>
    </div>
  </div>

  <!-- Per-run histograms (tabbed) -->
  <div class="section">
    <h2>Recall Distributions (per Run)</h2>
    <div class="tab-bar">{tab_buttons}</div>
    {hist_panels}
  </div>

</div>
<script>
{_CHART_DEFAULTS_JS}

const runNames = {run_names_json};
const p01Data = {p01_json};
const p05Data = {p05_json};
const p10Data = {p10_json};
const p50Data = {p50_json};
const qpsData = {qps_json};
const histData = {hist_data_json};

new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{
    labels: runNames,
    datasets: [
      {{ label: 'p1 recall', data: p01Data, borderColor: '#ef4444', backgroundColor: 'rgba(239,68,68,0.1)', tension: 0.3, pointRadius: 5 }},
      {{ label: 'p5 recall', data: p05Data, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', tension: 0.3, pointRadius: 5 }},
      {{ label: 'p10 recall', data: p10Data, borderColor: '#eab308', backgroundColor: 'rgba(234,179,8,0.1)', tension: 0.3, pointRadius: 5 }},
      {{ label: 'p50 recall', data: p50Data, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', tension: 0.3, pointRadius: 5 }},
    ],
  }},
  options: {{
    ...chartDefaults,
    plugins: {{
      ...chartDefaults.plugins,
      {stop_annotation_js}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e2433' }} }},
      y: {{ min: 0, max: 100, ticks: {{ color: '#64748b', callback: v => v + '%' }}, grid: {{ color: '#2d3748' }},
           title: {{ display: true, text: 'Recall (%)', color: '#64748b' }} }},
    }},
  }},
}});

new Chart(document.getElementById('qpsChart'), {{
  type: 'bar',
  data: {{
    labels: runNames,
    datasets: [{{
      label: 'ANN QPS', data: qpsData,
      backgroundColor: 'rgba(56,189,248,0.7)', borderColor: '#38bdf8', borderWidth: 1,
    }}],
  }},
  options: {{
    ...chartDefaults,
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e2433' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#2d3748' }},
           title: {{ display: true, text: 'QPS', color: '#64748b' }} }},
    }},
  }},
}});

histData.forEach((run, i) => makeHistChart(`hist-${{i}}`, run));

function showHistTab(i) {{
  document.querySelectorAll('.hist-panel').forEach((el, j) => el.classList.toggle('active', j === i));
  document.querySelectorAll('.tab-btn').forEach((el, j) => el.classList.toggle('active', j === i));
}}
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)
    print(f"\n✅ Sweep HTML report saved to: {output_path}")
