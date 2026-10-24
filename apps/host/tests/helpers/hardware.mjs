/** A roomy machine for tests about routing and lifecycle, whatever the memory of the PC running them. */
export const ROOMY_HOST = JSON.stringify({
  totalGb: 256,
  freeGb: 200,
  gpus: [
    { id: 'GPU-test-0', index: 0, sensorIndex: 0, vendor: 'nvidia', name: 'NVIDIA RTX 6000 Test', vramGb: 96, unified: false, backend: 'cuda', driver: '999.0' },
    { id: 'GPU-test-1', index: 1, sensorIndex: 1, vendor: 'nvidia', name: 'NVIDIA RTX 6000 Test', vramGb: 96, unified: false, backend: 'cuda', driver: '999.0' },
  ],
});
