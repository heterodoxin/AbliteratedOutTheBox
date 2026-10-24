import { tr } from '../lib/i18n';
import type { CSSProperties } from 'react';
import { Icon } from './ui';
/** Illustrates the selected budget; these are preferences, not live utilisation readings. */
export function OnboardingResources({ profile }: {
    profile: 'work' | 'performance';
}) {
    const work = profile === 'work', share = work ? 80 : 100;
    return <section className="onboard-sharing" data-profile={profile} aria-label={tr("Resource preference preview")} style={{ '--strata-share': `${share}%` } as CSSProperties}>
  <div className="onboard-sharing-computer" aria-hidden="true">
   <div className="onboard-sharing-screen">
    <div className="onboard-sharing-toolbar"><span /><span /><span /><small>{tr("Your computer")}</small></div>
    <div className="onboard-sharing-panes">
     <div className="onboard-sharing-model"><span className="onboard-sharing-app-label"><Icon name="models" size={16}/>{tr("Strata")}<span className="onboard-sharing-dot"/></span><div className="onboard-sharing-document"><Icon name="file" size={21}/><div><i /><i /><i /></div></div><div className="onboard-sharing-tokens"><i /><i /><i /><i /><i /></div></div>
     <div className="onboard-sharing-apps"><span>{tr("Other apps")}</span><div><Icon name="search" size={17}/></div><div><Icon name="file" size={17}/></div></div>
    </div>
   </div><div className="onboard-sharing-base"/>
  </div>
  <div className="onboard-sharing-budget">
   <div className="onboard-sharing-caption"><Icon name={work ? 'computer' : 'bolt'} size={16}/>{tr(work ? 'Room for your everyday work' : 'Full priority for Strata')}</div>
   <div className="onboard-sharing-numbers"><div><strong>{tr(share)}<small>%</small></strong><span>{tr("Strata\u2019s budget")}</span></div><div className="onboard-sharing-headroom"><strong>{tr(work ? '20' : '0')}<small>%</small></strong><span>{tr("Extra headroom")}</span></div></div>
   <div className="onboard-sharing-bar" aria-hidden="true"><span /><i /></div>
   <p>{tr(work ? 'Keep your browser and tools alongside your AI.' : 'Give your AI the available resources for focused work.')}</p>
  </div>
 </section>;
}
