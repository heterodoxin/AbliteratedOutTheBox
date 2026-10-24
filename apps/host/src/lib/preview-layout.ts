export const PREVIEW_SPLIT_KEY='strata.preview-split.v1';
export const DEFAULT_PREVIEW_FRACTION=1.15/2.15;
export const MIN_CHAT_WIDTH=360, MIN_PREVIEW_WIDTH=320;
export function previewWidth(total:number,fraction:number){
 if(!Number.isFinite(total)||total<=0)return 0;
 const lower=Math.min(MIN_PREVIEW_WIDTH,total/2),upper=Math.max(lower,total-MIN_CHAT_WIDTH);
 const value=Number.isFinite(fraction)?fraction:DEFAULT_PREVIEW_FRACTION;
 return Math.round(Math.max(lower,Math.min(upper,total*value)));
}
export function pointerPreviewFraction(total:number,right:number,x:number){return total>0?previewWidth(total,(right-x)/total)/total:DEFAULT_PREVIEW_FRACTION;}
