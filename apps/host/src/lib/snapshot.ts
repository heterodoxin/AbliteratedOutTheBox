/** Capture a complete view, including external SVG sprite symbols. */
export async function monitorPng(element: HTMLElement) {
  await document.fonts.ready;
  const [{ toPng }, spriteResponse] = await Promise.all([import('html-to-image'), fetch('/icons/sprite.svg')]);
  if (!spriteResponse.ok) throw new Error('Screenshot icons could not be loaded. Try again.');
  const sprite = new DOMParser().parseFromString(await spriteResponse.text(), 'image/svg+xml');
  const clone = element.cloneNode(true) as HTMLElement;
  clone.style.width = `${element.getBoundingClientRect().width}px`;
  clone.style.position = 'fixed'; clone.style.left = '-100000px'; clone.style.top = '0';
  clone.setAttribute('aria-hidden', 'true');
  // What is only live state (the model state and the conversation cache) is not part of the picture.
  clone.querySelectorAll('[data-snapshot-skip]').forEach(node => node.remove());
  for (const use of clone.querySelectorAll('svg use')) {
    const id = use.getAttribute('href')?.split('#')[1];
    const symbol = id && sprite.getElementById(id), svg = use.parentElement;
    if (!symbol || !svg) continue;
    const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    for (const attr of symbol.attributes) {
      if (attr.name === 'viewBox') svg.setAttribute(attr.name, attr.value);
      else if (attr.name !== 'id') group.setAttribute(attr.name, attr.value);
    }
    for (const child of symbol.childNodes) group.appendChild(child.cloneNode(true));
    use.replaceWith(group);
  }
  document.body.appendChild(clone);
  try {
    return await toPng(clone, { pixelRatio: 2, backgroundColor: getComputedStyle(document.documentElement).getPropertyValue('--st-surface').trim(), style: { position: 'static', left: 'auto', top: 'auto', margin: '0' } });
  } finally { clone.remove(); }
}
