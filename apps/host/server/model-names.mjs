/** The name a person reads for a loaded model: the family and the version from the catalog ("Qwen3.8-Flash-Next Q2_0"), not the file id. */
export function friendlyModelName(catalog, id) {
  for (const family of catalog?.families ?? []) {
    const size = (family.sizes ?? []).find(s => s.id === id);
    if (size?.model && family.title) return `${String(family.title).replace(/\s*\(Gyro\)\s*$/, '')} ${size.model}`.trim();
  }
  return null;
}
