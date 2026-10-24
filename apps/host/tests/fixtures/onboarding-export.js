/* Local screenshot fixture only. Export controls sit outside the captured app. */
(async()=>{
 await document.fonts.ready;
 const button=document.createElement('button');
 button.textContent='Export full-quality PNG';
 button.style.cssText='position:absolute;right:16px;top:16px;z-index:2147483647;padding:12px 18px;border:1px solid #047857;border-radius:12px;background:white;color:#171717;font:600 14px system-ui;cursor:pointer';
 button.dataset.fixtureCapture='true';
 const placeButton=()=>{const modal=document.querySelector('.setupmodal__surface');if(modal&&button.parentElement!==modal){modal.append(button);button.inert=false;}};
 document.body.append(button);placeButton();
 new MutationObserver(placeButton).observe(document.getElementById('root'),{childList:true,subtree:true});
 button.addEventListener('click',async()=>{
  button.disabled=true;button.textContent='Rendering full-quality PNG…';
  const frozen=document.createElement('style');
  frozen.textContent='#root *,#root *::before,#root *::after{animation-play-state:paused!important;transition:none!important}';
  document.head.append(frozen);
  const restoredIcons=[];
  try{
   await document.fonts.ready;
   // SVG foreignObject exports cannot resolve external <use> references.
   // Embed the app's original vector paths, retaining the same viewBox and fill.
   const spriteText=await (await fetch('/icons/sprite.svg')).text();
   const sprite=new DOMParser().parseFromString(spriteText,'image/svg+xml');
   for(const use of document.querySelectorAll('#root .st-icon use')){
    const symbol=sprite.getElementById(use.getAttribute('href').split('#')[1]);
    if(!symbol)throw new Error('An app icon could not be embedded');
    const svg=use.closest('svg'),viewBox=svg.getAttribute('viewBox'),transform=svg.style.transform;
    // Keep transforms from selectors that refer to the original <use>.
    svg.style.transform=getComputedStyle(svg).transform;
    const group=document.createElementNS('http://www.w3.org/2000/svg','g');
    group.setAttribute('fill',symbol.getAttribute('fill')??'none');
    for(const child of symbol.children)group.append(document.importNode(child,true));
    svg.setAttribute('viewBox',symbol.getAttribute('viewBox'));use.replaceWith(group);
    restoredIcons.push({svg,viewBox,transform,use,group});
   }
   const width=innerWidth,height=innerHeight,scale=4;
   const dataUrl=await htmlToImage.toPng(document.getElementById('root'),{
    width,height,pixelRatio:scale,skipAutoScale:true,preferredFontFormat:'woff2',
    backgroundColor:getComputedStyle(document.body).backgroundColor,
    filter:node=>node.dataset?.fixtureCapture!=='true',
   });
   const response=await fetch('/__fixture/capture',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({dataUrl,width:width*scale,height:height*scale})});
   const result=await response.json();if(!response.ok)throw new Error(result.error);
   button.textContent=`PNG saved · ${result.width} × ${result.height}`;
  }catch(error){button.textContent=`Export failed: ${error.message}`;}
  finally{
   for(const {svg,viewBox,transform,use,group} of restoredIcons){group.replaceWith(use);svg.style.transform=transform;if(viewBox===null)svg.removeAttribute('viewBox');else svg.setAttribute('viewBox',viewBox);}
   button.disabled=false;frozen.remove();
  }
 });
})();
