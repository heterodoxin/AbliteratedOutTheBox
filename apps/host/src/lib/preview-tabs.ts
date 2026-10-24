/** Pick the adjacent tab when the selected preview closes. */
export function closePreviewTab<T extends {id:string}>(items:T[], selected:string|null, id:string) {
  const index=items.findIndex(item=>item.id===id);
  const remaining=items.filter(item=>item.id!==id);
  return {items:remaining,selected:selected===id?(remaining[Math.min(index,remaining.length-1)]?.id??null):selected};
}
