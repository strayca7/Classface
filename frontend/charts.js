// charts.js — Chart.js initializations for the defense demo page
// All canvas IDs must match index.html

Chart.defaults.font.family = "'PingFang SC','Microsoft YaHei','Inter',sans-serif";
Chart.defaults.font.size = 12;
Chart.defaults.color = '#64748b';

const BLUE  = '#2563eb';
const SKY   = '#0ea5e9';
const GRN   = '#059669';
const ORG   = '#d97706';
const RED   = '#dc2626';
const LGRAY = '#e2e8f0';
const tick  = { grid: { color: '#f1f5f9' } };

// ── cOverview: horizontal bar — A/B/C accuracy overview ──
new Chart(document.getElementById('cOverview'), {
  type: 'bar',
  data: {
    labels: ['A 基线（干净）', 'B 遮挡 Naive', 'C v1 (L1=0.8)', 'C v2 (L1=0.5)', 'C v3 (禁用L2)'],
    datasets: [{
      label: 'Top-1 准确率 (%)',
      data: [92.65, 89.11, 12.52, 48.35, 89.11],
      backgroundColor: [GRN, SKY, RED, ORG, GRN],
      borderRadius: 6,
      borderSkipped: false,
    }],
  },
  options: {
    indexAxis: 'y',
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ' ' + c.parsed.x + '%' } },
    },
    scales: {
      x: { ...tick, min: 0, max: 100, ticks: { callback: v => v + '%' } },
      y: { grid: { display: false } },
    },
  },
});

// ── cTrain: dual-axis line — Train Loss (left) + Val Dice (right) ──
new Chart(document.getElementById('cTrain'), {
  type: 'line',
  data: {
    labels: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
    datasets: [
      {
        label: 'Train Loss',
        data: [0.3932,0.2814,0.2103,0.1912,0.1742,0.1355,0.1108,0.0993,
               0.0917,0.0885,0.0821,0.0756,0.0692,0.0621,0.0554,0.0512,
               0.0481,0.0470,0.0458,0.0455],
        borderColor: RED,
        backgroundColor: 'rgba(220,38,38,.08)',
        tension: 0.4,
        yAxisID: 'y',
        pointRadius: 0,
      },
      {
        label: 'Val Dice',
        data: [0.8454,0.8501,0.8530,0.8580,0.8648,0.8682,0.8695,0.8701,
               0.8712,0.8630,0.8648,0.8670,0.8690,0.8701,0.8715,0.8720,
               0.8729,0.8718,0.8705,0.8695],
        borderColor: BLUE,
        backgroundColor: 'rgba(37,99,235,.08)',
        tension: 0.4,
        yAxisID: 'y2',
        pointRadius: (ctx) => ctx.dataIndex === 16 ? 6 : 0,
        pointBackgroundColor: (ctx) => ctx.dataIndex === 16 ? GRN : BLUE,
      },
    ],
  },
  options: {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { position: 'top' },
      tooltip: {},
    },
    scales: {
      y:  { ...tick, title: { display: true, text: 'Loss' }, position: 'left' },
      y2: { ...tick, title: { display: true, text: 'Val Dice' }, position: 'right',
            min: 0.83, max: 0.88, grid: { display: false } },
      x:  { ...tick, title: { display: true, text: 'Epoch' } },
    },
  },
});

// ── cFg: bar — foreground ratio per method ──
new Chart(document.getElementById('cFg'), {
  type: 'bar',
  data: {
    labels: ['YCrCb', 'GMM', 'GrabCut', 'Watershed', 'ResUNet'],
    datasets: [{
      label: '前景占比 (%)',
      data: [54.7, 93.7, 28.9, 31.3, 28.9],
      backgroundColor: [ORG, RED, SKY, ORG, GRN],
      borderRadius: 6,
      borderSkipped: false,
    }],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ' ' + c.parsed.y + '%' } },
    },
    scales: {
      y: { ...tick, max: 100, ticks: { callback: v => v + '%' } },
      x: { grid: { display: false } },
    },
  },
});

// ── cIoU: bar — IoU vs GrabCut (GrabCut omitted as reference) ──
new Chart(document.getElementById('cIoU'), {
  type: 'bar',
  data: {
    labels: ['YCrCb', 'GMM', 'Watershed', 'ResUNet'],
    datasets: [{
      label: 'IoU vs GrabCut',
      data: [0.499, 0.308, 0.254, 0.855],
      backgroundColor: [ORG, RED, ORG, GRN],
      borderRadius: 6,
      borderSkipped: false,
    }],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ' IoU: ' + c.parsed.y } },
    },
    scales: {
      y: { ...tick, min: 0, max: 1 },
      x: { grid: { display: false } },
    },
  },
});

// ── cScale: line — keypoint detection rate vs input size (upsample strategy) ──
new Chart(document.getElementById('cScale'), {
  type: 'line',
  data: {
    labels: ['112px', '160px', '224px', '256px', '320px', '384px'],
    datasets: [{
      label: '关键点检测率 (%)',
      data: [12, 34, 61, 78, 96.9, 97.2],
      borderColor: BLUE,
      backgroundColor: 'rgba(37,99,235,.1)',
      tension: 0.35,
      fill: true,
      pointRadius: 5,
      pointBackgroundColor: [LGRAY, LGRAY, LGRAY, LGRAY, GRN, LGRAY],
      pointBorderColor:     ['#94a3b8','#94a3b8','#94a3b8','#94a3b8', GRN,'#94a3b8'],
      pointBorderWidth: 2,
    }],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ' ' + c.parsed.y + '%' } },
    },
    scales: {
      y: { ...tick, min: 0, max: 100, ticks: { callback: v => v + '%' } },
      x: { grid: { display: false }, title: { display: true, text: '输入分辨率' } },
    },
  },
});

// ── cSynth: doughnut — 3 occlusion types equal 7484 each ──
new Chart(document.getElementById('cSynth'), {
  type: 'doughnut',
  data: {
    labels: ['水杯 cup', '普通眼镜 glasses', '墨镜 sunglasses'],
    datasets: [{
      data: [7484, 7484, 7484],
      backgroundColor: [SKY, BLUE, ORG],
      borderWidth: 0,
      hoverOffset: 8,
    }],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'bottom' },
      tooltip: {
        callbacks: {
          label: c => ' ' + c.label + ': ' + c.parsed.toLocaleString() + ' 张',
        },
      },
    },
  },
});

// ── cIter: bar — cascade iteration results B/v1/v2/v3 ──
new Chart(document.getElementById('cIter'), {
  type: 'bar',
  data: {
    labels: ['B Naive', 'C v1 (L1=0.8)', 'C v2 (L1=0.5)', 'C v3 (禁用L2)'],
    datasets: [{
      label: '整体准确率 (%)',
      data: [89.11, 12.52, 48.35, 89.11],
      backgroundColor: [GRN, RED, ORG, GRN],
      borderRadius: 8,
      borderSkipped: false,
    }],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: c => ' ' + c.parsed.y + '%' } },
    },
    scales: {
      y: { ...tick, min: 0, max: 100, ticks: { callback: v => v + '%' } },
      x: { grid: { display: false } },
    },
  },
});

// ── cPerType: grouped bar — B vs C-v2 per occlusion type ──
new Chart(document.getElementById('cPerType'), {
  type: 'bar',
  data: {
    labels: ['水杯', '眼镜', '墨镜'],
    datasets: [
      {
        label: 'B Naive',
        data: [87.72, 90.69, 88.79],
        backgroundColor: GRN,
        borderRadius: 6,
      },
      {
        label: 'C v2',
        data: [42.04, 54.52, 48.48],
        backgroundColor: ORG,
        borderRadius: 6,
      },
    ],
  },
  options: {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      tooltip: { callbacks: { label: c => ' ' + c.dataset.label + ': ' + c.parsed.y + '%' } },
    },
    scales: {
      y: { ...tick, min: 0, max: 100, ticks: { callback: v => v + '%' } },
      x: { grid: { display: false } },
    },
  },
});

// ── cSummary: radar — 5 dimensions summary ──
new Chart(document.getElementById('cSummary'), {
  type: 'radar',
  data: {
    labels: ['基线准确率', '遮挡鲁棒性', '分割精度(Dice)', '关键点成功率', '数据规模'],
    datasets: [
      {
        label: '本项目',
        data: [92.65, 89.11, 87.29, 96.9, 75],
        borderColor: BLUE,
        backgroundColor: 'rgba(37,99,235,.12)',
        pointBackgroundColor: BLUE,
        pointRadius: 4,
      },
      {
        label: '无优化下界',
        data: [92.65, 89.11, 0, 96.9, 75],
        borderColor: LGRAY,
        backgroundColor: 'rgba(226,232,240,.2)',
        pointBackgroundColor: LGRAY,
        pointRadius: 3,
      },
    ],
  },
  options: {
    responsive: true,
    scales: {
      r: {
        ticks: { display: false },
        grid: { color: '#e2e8f0' },
        pointLabels: { font: { size: 12 } },
      },
    },
    plugins: { legend: { position: 'top' } },
  },
});

// ── Fade-in on scroll (IntersectionObserver: .fu -> .vis) ──
const fadeObserver = new IntersectionObserver(
  entries => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('vis'); }),
  { threshold: 0.12 }
);
document.querySelectorAll('.fu').forEach(el => fadeObserver.observe(el));

// ── Nav sidebar highlight on scroll ──
const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.sidebar-link');

const navObserver = new IntersectionObserver(
  entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        navLinks.forEach(l => {
          const active = l.getAttribute('href') === '#' + e.target.id;
          l.classList.toggle('active', active);
        });
      }
    });
  },
  { threshold: 0.3 }
);
sections.forEach(s => navObserver.observe(s));
