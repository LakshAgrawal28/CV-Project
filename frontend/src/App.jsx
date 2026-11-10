import React, { useState, useEffect, useMemo } from 'react';
import { 
  Upload, Cpu, Zap, Activity, Image as ImageIcon, Sparkles, Layers, 
  Sliders, ArrowRight, CheckCircle2, RefreshCw, BarChart3, TrendingUp, 
  AlertCircle, Filter, Eye, ChevronRight, Award, Compass, PieChart
} from 'lucide-react';
import benchmarkSummaryData from './benchmark_summary.json';

const DIAGRAMS = [
  {
    id: 'bars',
    title: '1. Benchmark Comparison Bars',
    subtitle: 'Full-reference (PSNR, SSIM, LPIPS) & Non-reference (UIQM, UCIQE) with Error Bars',
    src: '/visualizations/1_quantitative_benchmark_bars.png',
    insight: 'Proposed model improves average PSNR from 22.66 dB to 23.72 dB (+1.06 dB) and cuts perceptual distance LPIPS by 12.6%.'
  },
  {
    id: 'waterfall',
    title: '2. Per-Sample Win/Loss Waterfall',
    subtitle: 'Diverging Bar Plot of Δ PSNR (Proposed − Baseline) across all 46 test images',
    src: '/visualizations/2_win_loss_delta_distribution.png',
    insight: 'Proposed method wins on 32/46 samples (up to +3.38 dB gain). In 14/46 samples (coral & shallow scenes), baseline exceeded proposed by 0.35 - 1.32 dB.'
  },
  {
    id: 'scatter',
    title: '3. Head-to-Head Parity Scatter',
    subtitle: 'Fidelity & Structural Correlation with y = x Equality Boundary',
    src: '/visualizations/3_head_to_head_scatter.png',
    insight: 'Points above the diagonal indicate Proposed superior performance. Baseline retains clustering edge on high-brightness samples.'
  },
  {
    id: 'breakdown',
    title: '4. Optical Scenario Breakdown',
    subtitle: 'Grouped performance across 5 specific underwater degradation archetypes',
    src: '/visualizations/4_degradation_breakdown.png',
    insight: 'Proposed method dominates Severe Green (+2.53 dB) and Turbidity (+1.57 dB), while Baseline leads in Shallow/Balanced Light (-0.76 dB).'
  },
  {
    id: 'boxplot',
    title: '5. Statistical Box-Plots',
    subtitle: 'Median, Interquartile Ranges (IQR), and Sample Scatter Jitter',
    src: '/visualizations/5_metric_distributions_boxplot.png',
    insight: 'Proposed method lifts the lower 25th percentile of PSNR from 21.1 dB to 22.8 dB, significantly elevating worst-case performance.'
  },
  {
    id: 'radar',
    title: '6. Multi-Objective Quality Radar',
    subtitle: 'Trade-off analysis across Fidelity, Structure, Perceptual, Quality, and Chroma',
    src: '/visualizations/6_radar_tradeoff.png',
    insight: 'Balanced polygon expansion showing simultaneous gains in edge sharpness, color restoration, and contrast.'
  }
];

export default function App() {
  const [activeMainView, setActiveMainView] = useState('benchmarks'); // 'benchmarks' or 'inference'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState(null);
  const [baselineRes, setBaselineRes] = useState(null);
  const [proposedRes, setProposedRes] = useState(null);
  
  // Benchmark View States
  const [selectedDiagram, setSelectedDiagram] = useState(DIAGRAMS[0]);
  const [tableFilter, setTableFilter] = useState('all'); // 'all', 'proposed', 'baseline', 'green', 'shallow'
  const [searchTerm, setSearchTerm] = useState('');
  const [pairedData, setPairedData] = useState([]);

  useEffect(() => {
    fetchHealth();
    fetchPairedData();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch('/api/health');
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(data);
      }
    } catch (err) {
      console.warn("Backend not running at /api/health yet", err);
    }
  };

  const fetchPairedData = async () => {
    try {
      const res = await fetch('/paired_comparison.csv');
      if (res.ok) {
        const text = await res.text();
        const rows = text.trim().split('\n');
        const headers = rows[0].split(',');
        const parsed = rows.slice(1).map(line => {
          const vals = line.split(',');
          const obj = {};
          headers.forEach((h, i) => {
            const val = vals[i];
            obj[h.trim()] = isNaN(val) ? val : parseFloat(val);
          });
          return obj;
        });
        setPairedData(parsed);
      }
    } catch (err) {
      console.warn("Failed to load paired comparison data", err);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setBaselineRes(null);
      setProposedRes(null);
    }
  };

  const handleRunEnhancement = async () => {
    if (!selectedFile) return;
    setLoading(true);

    try {
      const formBase = new FormData();
      formBase.append('file', selectedFile);
      const resBase = await fetch('/api/enhance/baseline', {
        method: 'POST',
        body: formBase
      });
      const dataBase = await resBase.json();
      setBaselineRes(dataBase);

      const formProp = new FormData();
      formProp.append('file', selectedFile);
      const resProp = await fetch('/api/enhance/proposed', {
        method: 'POST',
        body: formProp
      });
      const dataProp = await resProp.json();
      setProposedRes(dataProp);
    } catch (err) {
      alert("Error enhancing image: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  // Filtered Paired Samples
  const filteredSamples = useMemo(() => {
    return pairedData.filter(item => {
      // Search
      const matchesSearch = item.image ? item.image.toLowerCase().includes(searchTerm.toLowerCase()) : true;
      if (!matchesSearch) return false;

      if (tableFilter === 'proposed') return item.delta_psnr > 0;
      if (tableFilter === 'baseline') return item.delta_psnr <= 0;
      if (tableFilter === 'green') return item.scenario?.includes('Green');
      if (tableFilter === 'shallow') return item.scenario?.includes('Shallow');
      if (tableFilter === 'turbidity') return item.scenario?.includes('Turbidity');
      return true;
    });
  }, [pairedData, tableFilter, searchTerm]);

  const summary = benchmarkSummaryData;

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-100 flex flex-col selection:bg-cyan-500 selection:text-white">
      {/* Top Header */}
      <header className="border-b border-slate-800/80 glass-panel sticky top-0 z-50 px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-gradient-to-tr from-cyan-500 to-blue-600 rounded-xl shadow-lg shadow-cyan-500/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold bg-gradient-to-r from-white via-slate-100 to-cyan-400 bg-clip-text text-transparent">
                  AquaVision
                </h1>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  Research Prototype
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Degradation-Aware Underwater Enhancement vs UIEC2Net Baseline
              </p>
            </div>
          </div>

          {/* Navigation Mode Switcher */}
          <div className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveMainView('benchmarks')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeMainView === 'benchmarks'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Quantitative Benchmarks</span>
            </button>
            <button
              onClick={() => setActiveMainView('inference')}
              className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                activeMainView === 'inference'
                  ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white shadow-md shadow-cyan-500/20'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>Live Image Inference</span>
            </button>
          </div>

          {/* Hardware Status */}
          <div className="hidden md:flex items-center space-x-3 bg-slate-900/80 px-3.5 py-1.5 rounded-xl border border-slate-800 text-xs">
            <div className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-slate-400">Device:</span>
              <span className="font-semibold text-cyan-300">
                {backendStatus?.device_name || 'NVIDIA RTX (Auto-Detect)'}
              </span>
            </div>
            <div className="h-3 w-[1px] bg-slate-700"></div>
            <div className="flex items-center space-x-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="font-semibold text-emerald-400">Ready</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        
        {/* ================================================================ */}
        {/* VIEW 1: QUANTITATIVE BENCHMARKS & VISUAL DIAGRAMS (DEFAULT)       */}
        {/* ================================================================ */}
        {activeMainView === 'benchmarks' && (
          <div className="space-y-6">
            
            {/* Header Banner */}
            <section className="glass-panel rounded-2xl p-6 relative overflow-hidden border border-slate-800">
              <div className="absolute -right-20 -top-20 w-80 h-80 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>
              <div className="relative z-10 max-w-4xl space-y-2">
                <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 text-xs font-semibold border border-emerald-500/20">
                  <Award className="w-3.5 h-3.5" />
                  <span>Empirical Benchmark Results · 46 Test Images Evaluated</span>
                </div>
                <h2 className="text-2xl font-bold text-white">
                  Quantitative Enhancement Performance & Comparative Diagnostics
                </h2>
                <p className="text-slate-300 text-sm leading-relaxed">
                  Rigorous comparison between the static <strong>UIEC2Net baseline</strong> and our 
                  proposed <strong>Degradation-Aware FiLM Modulation</strong> across 5 optical underwater regimes.
                </p>
              </div>
            </section>

            {/* High-Impact Stat Cards */}
            <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* PSNR Card */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2 relative overflow-hidden">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold">Average PSNR (Fidelity)</span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-bold">
                    +{summary.deltas.psnr} dB
                  </span>
                </div>
                <div className="flex items-baseline space-x-3">
                  <span className="text-2xl font-black text-white">{summary.proposed_metrics.psnr_mean} dB</span>
                  <span className="text-xs text-slate-400">vs {summary.baseline_metrics.psnr_mean} dB baseline</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full rounded-full" style={{ width: '84%' }}></div>
                </div>
                <p className="text-[11px] text-slate-400">Higher indicates less reconstruction distortion</p>
              </div>

              {/* SSIM Card */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2 relative overflow-hidden">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold">Average SSIM (Structure)</span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-bold">
                    +{summary.deltas.ssim}
                  </span>
                </div>
                <div className="flex items-baseline space-x-3">
                  <span className="text-2xl font-black text-white">{summary.proposed_metrics.ssim_mean}</span>
                  <span className="text-xs text-slate-400">vs {summary.baseline_metrics.ssim_mean} baseline</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-500 to-emerald-400 h-full rounded-full" style={{ width: '88%' }}></div>
                </div>
                <p className="text-[11px] text-slate-400">Measures edge preservation & structural contour</p>
              </div>

              {/* LPIPS Card */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2 relative overflow-hidden">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold">Average LPIPS (Perceptual)</span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-bold">
                    -12.6%
                  </span>
                </div>
                <div className="flex items-baseline space-x-3">
                  <span className="text-2xl font-black text-white">{summary.proposed_metrics.lpips_mean}</span>
                  <span className="text-xs text-slate-400">vs {summary.baseline_metrics.lpips_mean} baseline</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-gradient-to-r from-emerald-400 to-cyan-500 h-full rounded-full" style={{ width: '82%' }}></div>
                </div>
                <p className="text-[11px] text-slate-400">Lower is better (AlexNet deep feature distance)</p>
              </div>

              {/* Head-to-Head Win Rate Card */}
              <div className="glass-panel p-5 rounded-2xl border border-slate-800 space-y-2 relative overflow-hidden">
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="font-semibold">Head-to-Head Split</span>
                  <span className="text-[11px] px-2 py-0.5 rounded-full bg-cyan-500/15 text-cyan-400 font-bold">
                    46 Samples
                  </span>
                </div>
                <div className="flex items-baseline space-x-3">
                  <span className="text-2xl font-black text-cyan-400">{summary.win_rate_proposed}%</span>
                  <span className="text-xs text-slate-400">Proposed Wins ({summary.proposed_wins})</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden flex">
                  <div className="bg-emerald-400 h-full" style={{ width: `${summary.win_rate_proposed}%` }}></div>
                  <div className="bg-rose-500 h-full" style={{ width: `${100 - summary.win_rate_proposed}%` }}></div>
                </div>
                <p className="text-[11px] text-slate-400">
                  Baseline exceeded on <strong className="text-rose-400 font-mono">{summary.baseline_exceeded}</strong> samples (30.4%)
                </p>
              </div>
            </section>

            {/* Research Insight Callout */}
            <section className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 flex items-start space-x-4">
              <div className="p-2.5 bg-cyan-500/10 rounded-xl text-cyan-400 shrink-0 mt-0.5">
                <Compass className="w-5 h-5" />
              </div>
              <div className="space-y-1 text-xs text-slate-300 leading-relaxed">
                <h4 className="font-bold text-white text-sm">Key Empirical Finding: Why Baseline Exceeded in Specific Cases</h4>
                <p>
                  The baseline performed decently overall (<strong>22.66 dB avg PSNR</strong>). In severe wavelength-selective absorption
                  (<strong>Severe Green Cast</strong> and <strong>Deep Oceanic Blue</strong>) and heavy turbidity, our proposed degradation-aware
                  method was dramatically better (up to <strong className="text-emerald-400">+3.38 dB PSNR</strong> and <strong className="text-emerald-400">+0.068 SSIM</strong>).
                </p>
                <p className="pt-1 text-slate-400">
                  However, in <strong className="text-rose-300">Shallow Water with Balanced Illumination</strong> (e.g. samples 10, 13, 20, 28, 33), 
                  the baseline's static dual-color fusion was already optimal. In those balanced conditions, the proposed degradation encoder 
                  slightly over-modulated colors, allowing the baseline to exceed proposed performance by <strong className="text-rose-400">0.35 to 1.32 dB</strong>.
                </p>
              </div>
            </section>

            {/* Visual Diagrams Gallery (The 6 figures) */}
            <section className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <PieChart className="w-4 h-4 text-cyan-400" />
                    <span>Visual Research Diagnostic Diagrams</span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Select any diagram below to inspect statistical distributions, waterfall deltas, and multi-objective tradeoffs.
                  </p>
                </div>
              </div>

              {/* Diagram Selector Tabs */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
                {DIAGRAMS.map((diag) => (
                  <button
                    key={diag.id}
                    onClick={() => setSelectedDiagram(diag)}
                    className={`p-2.5 rounded-xl border text-left text-xs transition-all flex flex-col justify-between ${
                      selectedDiagram.id === diag.id
                        ? 'bg-cyan-950/40 border-cyan-500/50 shadow-md shadow-cyan-500/10 text-white'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
                    }`}
                  >
                    <span className="font-bold line-clamp-1">{diag.title}</span>
                    <span className="text-[10px] text-slate-500 mt-1 line-clamp-1">{diag.id}</span>
                  </button>
                ))}
              </div>

              {/* Active Diagram Display */}
              <div className="bg-slate-950/90 rounded-2xl border border-slate-800/90 p-4 space-y-3">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                  <div>
                    <h4 className="text-sm font-bold text-white">{selectedDiagram.title}</h4>
                    <p className="text-xs text-slate-400">{selectedDiagram.subtitle}</p>
                  </div>
                  <div className="text-xs bg-cyan-500/10 text-cyan-300 px-3 py-1 rounded-lg border border-cyan-500/20 max-w-md">
                    💡 <strong>Finding:</strong> {selectedDiagram.insight}
                  </div>
                </div>

                <div className="w-full flex items-center justify-center p-2 rounded-xl bg-[#0f172a] border border-slate-800/60 overflow-hidden min-h-[350px]">
                  <img
                    src={selectedDiagram.src}
                    alt={selectedDiagram.title}
                    className="max-h-[620px] w-auto object-contain rounded-lg shadow-xl"
                  />
                </div>
              </div>
            </section>

            {/* Per-Sample Quantitative Results Table */}
            <section className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-bold text-white flex items-center space-x-2">
                    <Filter className="w-4 h-4 text-cyan-400" />
                    <span>Per-Sample Benchmark Evaluation Table</span>
                  </h3>
                  <p className="text-xs text-slate-400">
                    Showing {filteredSamples.length} of {pairedData.length} total benchmark test images
                  </p>
                </div>

                {/* Filter and Search */}
                <div className="flex flex-wrap items-center gap-2">
                  <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-800 text-xs">
                    {[
                      { id: 'all', label: 'All (46)' },
                      { id: 'proposed', label: 'Proposed Wins (32)' },
                      { id: 'baseline', label: 'Baseline Exceeded (14)' },
                      { id: 'green', label: 'Severe Green' },
                      { id: 'shallow', label: 'Shallow Water' }
                    ].map(f => (
                      <button
                        key={f.id}
                        onClick={() => setTableFilter(f.id)}
                        className={`px-2.5 py-1 rounded-md text-xs font-semibold transition-all ${
                          tableFilter === f.id
                            ? 'bg-cyan-500 text-white'
                            : 'text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {f.label}
                      </button>
                    ))}
                  </div>

                  <input
                    type="text"
                    placeholder="Search image (e.g. 10)..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="bg-slate-900 border border-slate-800 text-xs text-slate-200 px-3 py-1.5 rounded-lg focus:outline-none focus:border-cyan-500 w-36"
                  />
                </div>
              </div>

              {/* Data Table */}
              <div className="overflow-x-auto rounded-xl border border-slate-800 max-h-[420px]">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-900 text-slate-400 sticky top-0 z-10 border-b border-slate-800">
                    <tr>
                      <th className="p-3">Image Sample</th>
                      <th className="p-3">Degradation Regime</th>
                      <th className="p-3">Baseline PSNR</th>
                      <th className="p-3">Proposed PSNR</th>
                      <th className="p-3">Δ PSNR</th>
                      <th className="p-3">Baseline SSIM</th>
                      <th className="p-3">Proposed SSIM</th>
                      <th className="p-3">Δ SSIM</th>
                      <th className="p-3">Proposed LPIPS</th>
                      <th className="p-3 text-center">Outcome</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 bg-slate-950/40">
                    {filteredSamples.map((row, idx) => {
                      const isProposedWin = row.delta_psnr > 0;
                      return (
                        <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                          <td className="p-3 font-bold text-white">{row.image}</td>
                          <td className="p-3 text-slate-300">{row.scenario || 'Benchmark'}</td>
                          <td className="p-3 text-slate-300">{row.psnr_base?.toFixed(2)} dB</td>
                          <td className="p-3 font-semibold text-white">{row.psnr_prop?.toFixed(2)} dB</td>
                          <td className={`p-3 font-bold ${isProposedWin ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {isProposedWin ? `+${row.delta_psnr?.toFixed(2)}` : row.delta_psnr?.toFixed(2)} dB
                          </td>
                          <td className="p-3 text-slate-300">{row.ssim_base?.toFixed(3)}</td>
                          <td className="p-3 font-semibold text-white">{row.ssim_prop?.toFixed(3)}</td>
                          <td className={`p-3 font-bold ${row.delta_ssim > 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                            {row.delta_ssim > 0 ? `+${row.delta_ssim?.toFixed(3)}` : row.delta_ssim?.toFixed(3)}
                          </td>
                          <td className="p-3 text-cyan-300">{row.lpips_prop?.toFixed(3)}</td>
                          <td className="p-3 text-center">
                            {isProposedWin ? (
                              <span className="inline-block px-2.5 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-sans text-[11px] font-semibold border border-emerald-500/30">
                                Proposed Won
                              </span>
                            ) : (
                              <span className="inline-block px-2.5 py-0.5 rounded-full bg-rose-500/15 text-rose-400 font-sans text-[11px] font-semibold border border-rose-500/30">
                                Baseline Exceeded
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ================================================================ */}
        {/* VIEW 2: LIVE IMAGE INFERENCE & COMPARISON                         */}
        {/* ================================================================ */}
        {activeMainView === 'inference' && (
          <div className="space-y-6">
            
            {/* Inference Header */}
            <section className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-2">
              <h2 className="text-xl font-bold text-white flex items-center space-x-2">
                <Zap className="w-5 h-5 text-cyan-400" />
                <span>Single-Image Interactive Enhancement</span>
              </h2>
              <p className="text-xs text-slate-400">
                Upload any underwater image to process through both UIEC2Net Baseline and our Proposed Degradation-Aware model in real time.
              </p>
            </section>

            {/* Upload & Controls Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              
              {/* File Upload Box */}
              <div className="glass-panel rounded-2xl p-6 space-y-4 flex flex-col border border-slate-800">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                  <Upload className="w-4 h-4 text-cyan-400" />
                  <span>Input Underwater Image</span>
                </h3>

                <div 
                  className={`border-2 border-dashed rounded-xl p-4 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[200px] relative ${
                    selectedFile ? 'border-cyan-500/50 bg-cyan-950/10' : 'border-slate-700/80 hover:border-cyan-500/40 bg-slate-900/40'
                  }`}
                  onClick={() => document.getElementById('image-upload-input').click()}
                >
                  <input
                    id="image-upload-input"
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={handleFileChange}
                  />

                  {previewUrl ? (
                    <div className="w-full h-full flex flex-col items-center justify-center space-y-2">
                      <img src={previewUrl} alt="Preview" className="max-h-44 w-auto object-contain rounded-lg" />
                      <span className="text-[11px] text-cyan-400 font-semibold">Click to replace image</span>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      <div className="p-3 bg-slate-800 rounded-full w-fit mx-auto text-slate-400">
                        <ImageIcon className="w-7 h-7 text-cyan-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-slate-300">Click or drag underwater image here</p>
                        <p className="text-xs text-slate-500 mt-1">Supports PNG, JPG, JPEG, BMP</p>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleRunEnhancement}
                  disabled={!selectedFile || loading}
                  className={`w-full py-3.5 px-4 rounded-xl font-semibold text-sm flex items-center justify-center space-x-2 shadow-lg transition-all ${
                    !selectedFile || loading
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed'
                      : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-cyan-500/25'
                  }`}
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      <span>Processing on GPU...</span>
                    </>
                  ) : (
                    <>
                      <Zap className="w-4 h-4" />
                      <span>Run Baseline & Proposed Models</span>
                    </>
                  )}
                </button>
              </div>

              {/* Research Method Comparison Card */}
              <div className="glass-panel rounded-2xl p-6 lg:col-span-2 space-y-5 flex flex-col justify-between border border-slate-800">
                <div>
                  <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2 mb-3">
                    <Sliders className="w-4 h-4 text-cyan-400" />
                    <span>Architectural Overview</span>
                  </h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Baseline Box */}
                    <div className="bg-slate-900/60 p-4 rounded-xl border border-slate-800/80 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-slate-300">1. UIEC2Net Baseline</span>
                        <span className="text-[10px] bg-slate-800 text-slate-400 px-2 py-0.5 rounded-full">Pretrained</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Standard dual-color space (RGB + HSV) underwater enhancement model. Maps input $X \rightarrow Y$ using static confidence fusion.
                      </p>
                      <div className="text-[11px] font-mono text-cyan-400/90 pt-1">
                        Formula: Y = F(X)
                      </div>
                    </div>

                    {/* Proposed Box */}
                    <div className="bg-slate-900/60 p-4 rounded-xl border border-cyan-500/30 space-y-2 relative overflow-hidden">
                      <div className="absolute top-0 right-0 w-16 h-16 bg-cyan-500/10 rounded-bl-full pointer-events-none"></div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-bold text-cyan-300">2. Proposed Degradation-Aware</span>
                        <span className="text-[10px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded-full font-semibold">Our Contribution</span>
                      </div>
                      <p className="text-xs text-slate-400 leading-relaxed">
                        Extracts compact degradation indicators (color cast, turbidity, Laplacian variance) and modulates features via FiLM layers.
                      </p>
                      <div className="text-[11px] font-mono text-cyan-400/90 pt-1">
                        Formula: z = D(X), Y = F(X, z)
                      </div>
                    </div>
                  </div>
                </div>

                {/* Quick Metrics Bar if results available */}
                {proposedRes && (
                  <div className="bg-cyan-950/30 border border-cyan-500/30 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
                    <div className="flex items-center space-x-3">
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                      <div>
                        <p className="text-xs font-semibold text-slate-200">GPU Inference Complete</p>
                        <p className="text-[11px] text-slate-400">Processed in {proposedRes.inference_ms} ms on {proposedRes.device}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-6 text-xs font-mono">
                      <div>
                        <span className="text-slate-400">UIQM: </span>
                        <span className="font-bold text-cyan-300">{proposedRes.metrics.uiqm}</span>
                      </div>
                      <div>
                        <span className="text-slate-400">UCIQE: </span>
                        <span className="font-bold text-cyan-300">{proposedRes.metrics.uciqe}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Results Triple Gallery */}
            {previewUrl && (
              <section className="glass-panel rounded-2xl p-6 border border-slate-800 space-y-4">
                <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
                  <Eye className="w-4 h-4 text-cyan-400" />
                  <span>Enhancement Comparison Gallery</span>
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* 1. Original Input */}
                  <div className="space-y-3">
                    <div className="text-xs font-semibold text-slate-400 px-1">1. Original Input</div>
                    <div className="bg-slate-900 rounded-xl overflow-hidden border border-slate-800 aspect-square flex items-center justify-center p-2">
                      <img src={previewUrl} alt="Original Input" className="w-full h-full object-contain rounded-lg" />
                    </div>
                  </div>

                  {/* 2. Baseline UIEC2Net */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs font-semibold text-slate-300 px-1">
                      <span>2. UIEC2Net Baseline</span>
                      {baselineRes && <span className="text-cyan-400 font-mono text-[11px]">{baselineRes.inference_ms} ms</span>}
                    </div>
                    <div className="bg-slate-900 rounded-xl overflow-hidden border border-slate-800 aspect-square flex items-center justify-center relative p-2">
                      {baselineRes ? (
                        <img src={baselineRes.enhanced_b64} alt="Baseline Enhanced" className="w-full h-full object-contain rounded-lg" />
                      ) : (
                        <div className="text-center p-4">
                          <Activity className="w-6 h-6 text-slate-600 mx-auto mb-2" />
                          <span className="text-xs text-slate-500">Click Run to generate Baseline</span>
                        </div>
                      )}
                    </div>
                    {baselineRes && (
                      <div className="bg-slate-900/60 p-3 rounded-lg border border-slate-800 text-xs font-mono space-y-1">
                        <div className="flex justify-between text-slate-400">
                          <span>UIQM (Quality):</span>
                          <span className="text-white font-semibold">{baselineRes.metrics.uiqm}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>UCIQE (Chroma):</span>
                          <span className="text-white font-semibold">{baselineRes.metrics.uciqe}</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 3. Proposed Method */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs font-semibold text-cyan-300 px-1">
                      <span>3. Proposed Method (Degradation-Aware)</span>
                      {proposedRes && <span className="text-cyan-400 font-mono text-[11px]">{proposedRes.inference_ms} ms</span>}
                    </div>
                    <div className="bg-slate-900 rounded-xl overflow-hidden border border-cyan-500/40 aspect-square flex items-center justify-center relative shadow-lg shadow-cyan-500/5 p-2">
                      {proposedRes ? (
                        <img src={proposedRes.enhanced_b64} alt="Proposed Enhanced" className="w-full h-full object-contain rounded-lg" />
                      ) : (
                        <div className="text-center p-4">
                          <Sparkles className="w-6 h-6 text-cyan-500/40 mx-auto mb-2" />
                          <span className="text-xs text-slate-500">Click Run to generate Proposed</span>
                        </div>
                      )}
                    </div>
                    {proposedRes && (
                      <div className="bg-cyan-950/30 p-3 rounded-lg border border-cyan-500/30 text-xs font-mono space-y-1">
                        <div className="flex justify-between text-slate-300">
                          <span>UIQM (Quality):</span>
                          <span className="text-cyan-300 font-bold">{proposedRes.metrics.uiqm}</span>
                        </div>
                        <div className="flex justify-between text-slate-300">
                          <span>UCIQE (Chroma):</span>
                          <span className="text-cyan-300 font-bold">{proposedRes.metrics.uciqe}</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </section>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-5 text-center text-xs text-slate-500">
        <p>AquaVision — Degradation-Aware Underwater Image Enhancement | UIEC2Net Baseline vs FiLM Feature Conditioning</p>
      </footer>
    </div>
  );
}
