/**
 * Hardware detection.
 *
 * Real probes, cross-platform, never fatal: if a command is missing we fall back
 * to Node's os module and a CPU-only verdict. The onboarding wizard shows exactly
 * what was found.
 */
import os from 'node:os';
import { readFile } from 'node:fs/promises';
import { parseCpuTopology, parseWindowsProcessors, parseNvidiaDevices, detectDrmGpus } from './device-probes.mjs';
import { CONTEXT_TOKENS, VARIANTS, footprint } from './catalog.mjs';
import { MODELS_DIR, diskFreeGb, diskTotalGb } from './store.mjs';
import { probe, round } from './util.mjs';
import { utf8Script } from './powershell-utf8.mjs';

let cached = null;
let cachedAt = 0;

export function invalidate() {
  cached = null;
}

/**
 * The last detected hardware, synchronously.
 *
 * Callers that build config (the expert param list, for one) are synchronous and
 * cannot await a detection pass, but they still need to know how many cores the
 * machine has. Returns null before the first detection, so callers must have a
 * sensible fallback rather than assuming a machine shape.
 */
export function cachedHardware() {
  return cached ?? null;
}

function testHardware() {
  const raw = process.env.STRATA_TEST_HARDWARE;
  if (!raw) return null;
  try { const value = JSON.parse(raw); return value && typeof value === 'object' ? value : null; } catch { return null; }
}

export async function detect({ force = false } = {}) {
  if (!force && cached && Date.now() - cachedAt < 30_000) return cached;

  const platform = process.platform;
  const arch = process.arch;
  // Tests only: STRATA_TEST_HARDWARE='{"totalGb":128,"freeGb":100,"gpus":[...]}' stands in for the probes, so a
  // test about routing or lifecycle does not depend on how much memory the machine running it has.
  const forced = testHardware();
  const totalGb = forced?.totalGb ?? os.totalmem() / 1024 ** 3;
  // os.freemem() ignores reclaimable pages, which reads as "0 GB free" on macOS.
  const freeGb = forced?.freeGb ?? (await availableMemoryGb(platform)) ?? os.freemem() / 1024 ** 3;

  const cpu = await detectCpu(platform);
  const gpus = forced?.gpus ?? await detectGpus(platform, totalGb);
  const accelerator = pickAccelerator(gpus, cpu, totalGb);

  const hw = {
    probedAt: new Date().toISOString(),
    host: {
      name: os.hostname().replace(/\.local$/, ''),
      platform: platformLabel(platform),
      release: os.release(),
      arch,
      app: process.version,
    },
    cpu,
    processors: cpu.processors,
    memory: {
      totalGb: round(totalGb / 1, 1),
      freeGb: round(freeGb / 1, 1),
      usedGb: round(totalGb - freeGb, 1),
      pressure: freeGb / totalGb > 0.35 ? 'low' : freeGb / totalGb > 0.18 ? 'medium' : 'high',
    },
    disk: {
      path: MODELS_DIR,
      totalGb: round(diskTotalGb(MODELS_DIR), 0),
      freeGb: round(diskFreeGb(MODELS_DIR), 0),
      modelsDir: MODELS_DIR,
    },
    gpus,
    accelerator,
    checks: [],
    verdict: { ok: true, headline: '', detail: '' },
    recommendedVariantId: '',
  };

  hw.checks = buildChecks(hw);
  hw.verdict = buildVerdict(hw);
  hw.recommendedVariantId = recommend(hw);

  cached = hw;
  cachedAt = Date.now();
  return hw;
}

/** Reclaimable memory, not just the free list. */
export async function availableMemoryGb(platform) {
  if(platform==='linux') {
    try { const raw=await readFile('/proc/meminfo','utf8'); const available=Number(/MemAvailable:\s+(\d+)/.exec(raw)?.[1]); return available>0?available/1024**2:null; } catch { return null; }
  }
  if (platform !== 'darwin') return null;
  const out = await probe('vm_stat', []);
  if (!out) return null;
  const pageSize = Number(/page size of (\d+)/.exec(out)?.[1] ?? 4096);
  const pages = (label) => Number(new RegExp(`Pages ${label}:\\s+(\\d+)`).exec(out)?.[1] ?? 0);
  const reclaimable = pages('free') + pages('inactive') + pages('speculative') + pages('purgeable');
  if (!reclaimable) return null;
  return (reclaimable * pageSize) / 1024 ** 3;
}

async function detectCpu(platform) {
  const logical = os.cpus(), threads = logical.length || 1;
  let model = logical[0]?.model?.trim() ?? 'Unknown processor', ghz=round((logical[0]?.speed??0)/1000,2), processors=[];
  if(platform==='darwin') {
    const [brand,physical,threadCount,packages]=await Promise.all([
      probe('sysctl',['-n','machdep.cpu.brand_string']),probe('sysctl',['-n','hw.physicalcpu']),probe('sysctl',['-n','hw.logicalcpu']),probe('sysctl',['-n','hw.packages'])
    ]);
    if(brand)model=brand;
    const cores=Number(physical),count=Number(packages),totalThreads=Number(threadCount)||threads;
    if(cores>0&&count>0&&cores%count===0&&totalThreads%count===0)processors=Array.from({length:count},(_,i)=>({id:`socket-${i}`,socket:String(i),model,cores:cores/count,threads:totalThreads/count,arch:process.arch,ghz,...(count===1?{logicalIds:logical.map((_,index)=>index)}:{})}));
  } else if(platform==='linux') {
    const raw=await probe('lscpu',['-b','-p=CPU,CORE,SOCKET']);
    const names=new Map();
    try { for(const block of (await readFile('/proc/cpuinfo','utf8')).split(/\n\s*\n/)) {
      const fields=Object.fromEntries(block.split('\n').filter(l=>l.includes(':')).map(l=>{const i=l.indexOf(':');return [l.slice(0,i).trim(),l.slice(i+1).trim()];}));
      if(fields['physical id']&&fields['model name'])names.set(fields['physical id'],fields['model name']);
    }}catch{}
    processors=parseCpuTopology(raw,model,process.arch,ghz,names);
  } else if(platform==='win32') {
    const raw=await probe('powershell.exe',['-NoProfile','-NonInteractive','-Command',utf8Script('Get-CimInstance Win32_Processor | Select-Object DeviceID,SocketDesignation,Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress')]);
    processors=parseWindowsProcessors(raw,process.arch);
  }
  const names=[...new Set(processors.map(p=>p.model))];
  return {model:names.length===1?names[0]:names.length>1?names.join(' + '):model,cores:processors.length?processors.reduce((n,p)=>n+p.cores,0):threads,threads:processors.length?processors.reduce((n,p)=>n+p.threads,0):threads,arch:process.arch,ghz,processors,topologyKnown:!!processors.length};
}

async function detectGpus(platform, totalGb) {
  const gpus = [];

  if (platform === 'darwin') {
    const raw = await probe('system_profiler', ['SPDisplaysDataType', '-json'], 8000);
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        const list = parsed?.SPDisplaysDataType ?? [];
        list.forEach((g, i) => {
          const name = g.sppci_model ?? g._name ?? 'Apple GPU';
          if (!name || /display/i.test(name)) return;
          const cores = Number(g.sppci_cores) || 0;
          gpus.push({
            id: `metal:${i}`,
            index: i,
            vendor: /apple/i.test(name) ? 'apple' : /amd|radeon/i.test(name) ? 'amd' : /intel/i.test(name) ? 'intel' : 'unknown',
            name: cores ? `${name} (${cores}-core GPU)` : name,
            // Apple silicon shares system memory; Metal's default wired limit is ~75%.
            vramGb: /apple/i.test(name)&&process.arch==='arm64' ? round(totalGb * 0.75, 1) : (Number.parseFloat(String(g.sppci_vram??g.spdisplays_vram??'')) || 0) * (/MB/i.test(String(g.sppci_vram??g.spdisplays_vram??''))?1/1024:1),
            unified: /apple/i.test(name)&&process.arch==='arm64',
            backend: 'metal',
          });
        });
      } catch {
        /* fall through */
      }
    }
    if (!gpus.length && process.arch === 'arm64') {
      const chip = await probe('sysctl', ['-n', 'machdep.cpu.brand_string']);
      gpus.push({
        id: 'metal:0',
        index: 0,
        vendor: 'apple',
        name: chip || 'Apple silicon GPU',
        vramGb: round(totalGb * 0.75, 1),
        unified: true,
        backend: 'metal',
      });
    }
  }

  const smi = await probe('nvidia-smi', ['--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version','--format=csv,noheader,nounits']);
  gpus.push(...parseNvidiaDevices(smi));
  if(platform==='linux') {
    const drm=await detectDrmGpus();
    for(const gpu of drm) {
      const existing=gpus.find(g=>g.busId?.replace(/^00000000:/,'0000:')===gpu.busId);
      if(existing)existing.sensorPath=gpu.sensorPath;else gpus.push(gpu);
    }
  }
  if(platform==='win32') {
    const raw=await probe('powershell.exe',['-NoProfile','-NonInteractive','-Command',utf8Script('Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress')]);
    try {const parsed=JSON.parse(raw);for(const row of Array.isArray(parsed)?parsed:[parsed]) {
      if(!row?.Name||(/nvidia/i.test(row.Name)&&gpus.some(g=>g.vendor==='nvidia')))continue;
      const vendor=/amd|radeon/i.test(row.Name)?'amd':/intel/i.test(row.Name)?'intel':'unknown';
      gpus.push({id:String(row.PNPDeviceID),index:gpus.length,vendor,name:row.Name,vramGb:0,unified:false,backend:'vulkan'});
    }}catch{}
  }
  gpus.forEach((g,i)=>{g.index=i;});

  return gpus;
}

function pickAccelerator(gpus, cpu, totalGb) {
  const best = [...gpus]
    .filter((g) => g.backend !== 'cpu')
    .sort((a, b) => b.vramGb - a.vramGb)[0];
  if (best) {
    return {
      backend: best.backend,
      label: best.name,
      usableVramGb: best.vramGb,
      unified: best.unified,
    };
  }
  return {
    backend: 'cpu',
    label: `${cpu.model.split('@')[0].trim()} (CPU only)`,
    usableVramGb: round(totalGb * 0.5, 1),
    unified: true,
  };
}

function buildChecks(hw) {
  const checks = [];
  const accel = hw.accelerator;

  checks.push(
    accel.backend === 'cpu'
      ? {
          id: 'gpu',
          label: 'GPU and driver',
          state: 'warn',
          detail: 'No GPU accelerator found — Strata will run on the CPU',
          value: accel.label,
        }
      : {
          id: 'gpu',
          label: 'GPU and driver',
          state: 'pass',
          detail: 'Compatible GPU found',
          value: `${accel.label} · ${accel.backend}`,
        },
  );

  const total = hw.memory.totalGb;
  checks.push(
    total >= 32
      ? { id: 'memory', label: 'System memory', state: 'pass', detail: 'Enough for recommended model', value: `${Math.round(total)} GB` }
      : total >= 16
        ? { id: 'memory', label: 'System memory', state: 'pass', detail: 'Enough for the smallest model', value: `${Math.round(total)} GB` }
        : { id: 'memory', label: 'System memory', state: 'fail', detail: 'Below the 16 GB minimum', value: `${Math.round(total)} GB` },
  );

  const cores = hw.cpu.cores;
  checks.push(
    cores >= 8
      ? { id: 'cpu', label: 'CPU', state: 'pass', detail: 'Supported', value: `${cores} cores` }
      : cores >= 4
        ? { id: 'cpu', label: 'CPU', state: 'pass', detail: 'Supported, slower prompt reading', value: `${cores} cores` }
        : { id: 'cpu', label: 'CPU', state: 'warn', detail: 'Few cores — expect slow prompt reading', value: `${cores} cores` },
  );

  const free = hw.disk.freeGb;
  checks.push(
    free >= 60
      ? { id: 'storage', label: 'Free storage', state: 'pass', detail: 'Space available', value: `${Math.round(free)} GB free` }
      : free >= 25
        ? { id: 'storage', label: 'Free storage', state: 'pass', detail: 'Enough for one model', value: `${Math.round(free)} GB free` }
        : { id: 'storage', label: 'Free storage', state: 'warn', detail: 'Tight — pick the smallest variant', value: `${Math.round(free)} GB free` },
  );

  return checks;
}

function buildVerdict(hw) {
  const failed = hw.checks.filter((c) => c.state === 'fail');
  const warned = hw.checks.filter((c) => c.state === 'warn');
  if (failed.length) {
    return {
      ok: false,
      headline: 'This computer is below the minimum.',
      detail: `${failed.map((f) => f.label).join(' and ')} need attention before a model will fit.`,
    };
  }
  if (warned.length) {
    return {
      ok: true,
      headline: 'This computer can run Strata.',
      detail: `${warned.map((w) => w.label).join(' and ')} ${warned.length > 1 ? 'are' : 'is'} a limit — Strata will pick a lighter model.`,
    };
  }
  return {
    ok: true,
    headline: 'This computer can run Strata comfortably.',
    detail: 'Everything needed for the recommended model is present.',
  };
}

/**
 * Recommend the heaviest variant that still fits fast memory at the default
 * context, using the same footprint maths the tuning screen shows. When nothing
 * fits, fall back to the smallest tier rather than the biggest.
 */
function recommend(hw) {
  const usable = hw.accelerator.unified
    ? hw.memory.totalGb * 0.72
    : hw.accelerator.usableVramGb + hw.memory.totalGb * 0.35;
  const ctx = CONTEXT_TOKENS.recommended;

  const viable = VARIANTS.map((v) => ({ v, fp: footprint(v, ctx) }))
    .filter(({ fp }) => fp.totalGb <= usable)
    .sort((a, b) => b.fp.totalGb - a.fp.totalGb);

  if (viable.length) return viable[0].v.id;

  const smallest = [...VARIANTS].sort((a, b) => footprint(a, ctx).totalGb - footprint(b, ctx).totalGb)[0];
  return smallest?.id ?? VARIANTS[0].id;
}

function platformLabel(platform) {
  return platform === 'darwin' ? 'macOS' : platform === 'win32' ? 'Windows' : platform === 'linux' ? 'Linux' : platform;
}
