import { tr } from '../lib/i18n';
/** Screen chrome: a titled header pinned above a scrolling body. */
import type { ReactNode } from 'react';
import { cx } from '../lib/format';
import { Icon } from './ui';
export function Screen({ title, sub, actions, children, width = 'normal', bare, plainHeader, titleDetail, back, }: {
    title: ReactNode;
    sub?: ReactNode;
    actions?: ReactNode;
    children: ReactNode;
    width?: 'normal' | 'tight' | 'wide' | 'full';
    bare?: boolean;
    plainHeader?: boolean;
    titleDetail?: ReactNode;
    /** A way back to the list this screen was opened from, shown before the title (as the chat's navbar shows its own title). */
    back?: { label: string; onClick: () => void };
}) {
    return (<>
      <header className={cx("st-header main__head", plainHeader && "main__head--plain")}>
        <div className="main__titles">
          <div className="main__title-line">{tr(back ? <button type="button" className="main__back" onClick={back.onClick}><Icon name="chevron" size={14}/>{tr(back.label)}</button> : null)}<h1 className="main__title">{tr(title)}</h1>{tr(titleDetail ? <span className="main__title-detail">{tr(titleDetail)}</span> : null)}</div>
          {tr(sub ? <p className="main__sub">{tr(sub)}</p> : null)}
        </div>
        {tr(actions ? <div className="main__actions">{tr(actions)}</div> : null)}
      </header>
      <div className="main__body">
        {tr(bare ? (children) : (<div className={cx('wrap', width === 'tight' && 'wrap--tight', width === 'wide' && 'wrap--wide')} style={width === 'full' ? { maxWidth: 'none' } : undefined}>
            {tr(children)}
          </div>))}
      </div>
    </>);
}
export function SectionHead({ title, sub, right, }: {
    title: ReactNode;
    sub?: ReactNode;
    right?: ReactNode;
}) {
    return (<div className="section-toolbar" style={{ alignItems: 'flex-end', marginBottom: 12 }}>
      <div>
        <h2 className="h3">{tr(title)}</h2>
        {tr(sub ? <div className="small muted" style={{ marginTop: 2 }}>{tr(sub)}</div> : null)}
      </div>
      {tr(right)}
    </div>);
}
