import { tr } from '../lib/i18n';
/**
 * Strata primitives.
 *
 * These wrap the kit's own component classes (`st-*` from ui/components.css)
 * rather than inventing a parallel set, so the app and the design reference stay
 * the same system. Anything added here is a documented extension that only uses
 * kit tokens — see src/styles/app.css.
 */
import { createContext, useContext, useCallback, useEffect, useId, useRef, useState, type AriaAttributes, type CSSProperties, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { cx, userError } from '../lib/format';
import { plainFailure } from '../lib/plain-errors';
import { dismissActionError, dismissToast, holdToast, releaseToast, useStore } from '../lib/store';
import { isTopSurface, useSurfaceFocus } from '../lib/focus';
/* -------------------------------------------------------------------- icon */
export type IconName = 'pause' | 'play' | 'activity' | 'plus' | 'shield' | 'computer'
// Authored for the shell: see docs/design-system-notes.md.
 | 'panel' | 'search' | 'sliders' | 'folder' | 'folder-plus' | 'file' | 'ellipsis' | 'pencil' | 'pin' | 'x' | 'attach' | 'bolt' | 'branch' | 'chat' | 'check' | 'chevron' | 'clock' | 'copy' | 'cpu' | 'disk' | 'download' | 'error' | 'experts' | 'gauge' | 'gpu' | 'image' | 'info' | 'expand' | 'contract' | 'layers' | 'models' | 'skills' | 'engine' | 'family-qwen' | 'model-file' | 'link' | 'memory' | 'mic' | 'moon' | 'new-chat' | 'refresh' | 'send' | 'settings' | 'stop' | 'sun' | 'thermometer' | 'thinking' | 'tool' | 'trash' | 'warning';
export function Icon({ name, size, className, style, }: {
    name: IconName;
    size?: number;
    className?: string;
    style?: CSSProperties;
}) {
    return (<svg className={cx('st-icon', className)} style={{ width: size, height: size, ...style }} aria-hidden="true" focusable="false">
      <use href={`/icons/sprite.svg#i-${name}`}/>
    </svg>);
}
/* ------------------------------------------------------------------ button */
type BtnProps = AriaAttributes & {
    children?: ReactNode;
    variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    icon?: IconName;
    iconRight?: IconName;
    block?: boolean;
    title?: string;
    disabled?: boolean;
    active?: boolean;
    type?: 'button' | 'submit';
    onClick?: () => void;
    className?: string;
};
export function Btn({ children, variant = 'secondary', size = 'md', icon, iconRight, block, title, disabled, active, type = 'button', onClick, className, ...aria }: BtnProps) {
    return (<button {...aria} type={type} title={tr(title)} disabled={disabled} onClick={onClick} className={cx('st-btn', `st-btn--${variant}`, size === 'sm' && 'st-btn--sm', size === 'lg' && 'st-btn--lg', block && 'st-btn--block', active && 'st-btn--active', className)}>
      {tr(icon ? <Icon name={icon}/> : null)}
      {tr(children)}
      {tr(iconRight ? <Icon name={iconRight}/> : null)}
    </button>);
}
export function IconBtn({ name, title, onClick, disabled, active, size = 19, }: {
    name: IconName;
    title: string;
    onClick?: () => void;
    disabled?: boolean;
    /** Renders in the pressed state — used by the sidebar's search toggle. */
    active?: boolean;
    size?: number;
}) {
    return (<button type="button" className={cx('st-btn st-btn--icon', active && 'is-on')} aria-pressed={active} title={tr(title)} aria-label={tr(title)} onClick={onClick} disabled={disabled}>
      <Icon name={name} size={size}/>
    </button>);
}
/* ------------------------------------------------------------------- panel */
export function Panel({ children, className, pad, }: {
    children: ReactNode;
    className?: string;
    pad?: boolean;
    accent?: boolean;
}) {
    return <section className={cx('st-card', !pad && 'st-card--flush', className)}>{tr(children)}</section>;
}
export function PanelHead({ title, sub, right, icon, }: {
    title: ReactNode;
    sub?: ReactNode;
    right?: ReactNode;
    icon?: IconName;
}) {
    return (<header className="st-card__head">
      {tr(icon ? <Icon name={icon} size={18} className="st-card__headicon"/> : null)}
      <div className="col" style={{ minWidth: 0, flex: 1 }}>
        {/* A real h3 so the document has an outline below the h2 section heads;
            styled exactly as before. */}
        <h3 className="h4">{tr(title)}</h3>
        {tr(sub ? <div className="small muted">{tr(sub)}</div> : null)}
      </div>
      {tr(right)}
    </header>);
}
/* -------------------------------------------------------------------- pill */
export function Pill({ children, tone = 'plain', dot, badge, className, }: {
    children: ReactNode;
    tone?: 'plain' | 'accent' | 'info' | 'warn' | 'danger';
    dot?: boolean;
    /** `badge` renders the 24px badge treatment instead of the 30px status pill. */
    badge?: boolean;
    className?: string;
}) {
    const toneClass = {
        plain: undefined,
        accent: badge ? 'st-badge--generating' : 'st-pill--accent',
        info: badge ? 'st-badge--reading' : 'st-pill--info',
        warn: badge ? 'st-badge--queued' : 'st-pill--warn',
        danger: badge ? 'st-badge--error' : 'st-pill--danger',
    }[tone];
    return (<span className={cx(badge ? 'st-badge' : 'st-pill', toneClass, className)}>
      {tr(dot && !badge ? <span className="st-pill__dot"/> : null)}
      {tr(children)}
    </span>);
}
/* --------------------------------------------------------------------- bar */
export function Bar({ value, tone = 'accent', size, indeterminate, label = 'Progress', }: {
    value: number;
    tone?: 'accent' | 'info' | 'warn' | 'danger';
    size?: 'sm' | 'lg';
    indeterminate?: boolean;
    label?: string;
}) {
    return (<div className={cx('st-progress', size === 'sm' && 'st-progress--sm', size === 'lg' && 'st-progress--lg', indeterminate && 'st-progress--indeterminate')} data-tone={tone === 'accent' ? undefined : tone} role="progressbar" aria-label={tr(label)} aria-valuenow={indeterminate ? undefined : Math.round(value)} aria-valuemin={0} aria-valuemax={100}>
      <div className="st-progress__bar" style={{ width: `${Math.max(0, Math.min(100, value))}%` }}/>
    </div>);
}
/* ---------------------------------------------------------------- segments */
export function Seg<T extends string>({ value, options, onChange, ariaLabel, panelId, }: {
    value: T;
    /** `title` explains a label that has to stay short to fit the control. */
    options: {
        value: T;
        label: string;
        icon?: IconName;
        title?: string;
        disabled?: boolean;
    }[];
    onChange: (v: T) => void;
    ariaLabel?: string;
    /** Supply the associated panel ID for actual tabs; omit for segmented values. */
    panelId?: string;
}) {
    return (<div className="st-tabs" role={panelId ? 'tablist' : 'radiogroup'} aria-label={tr(ariaLabel)}>
      {tr(options.map((o) => (<button key={o.value} type="button" role={panelId ? 'tab' : 'radio'} id={panelId ? `${panelId}-tab-${o.value}` : undefined} className="st-tab" aria-checked={panelId ? undefined : value === o.value} aria-selected={panelId ? value === o.value : undefined} aria-controls={panelId} disabled={o.disabled} tabIndex={value === o.value ? 0 : -1} onKeyDown={e => {
                if (!['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key))
                    return;
                e.preventDefault();
                const available = options.map((item, index) => ({ item, index })).filter(({ item }) => !item.disabled);
                const at = available.findIndex(({ item }) => item === o);
                const nextAt = e.key === 'Home' ? 0 : e.key === 'End' ? available.length - 1 : (at + (e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1 : -1) + available.length) % available.length;
                const next = available[nextAt]?.index;
                if (next === undefined)
                    return;
                onChange(options[next].value);
                (e.currentTarget.parentElement?.children[next] as HTMLElement)?.focus();
            }} title={tr(o.title)} onClick={() => onChange(o.value)}>
          {tr(o.icon ? <Icon name={o.icon}/> : null)}{tr(o.label)}
        </button>)))}
    </div>);
}
/* ------------------------------------------------------------------- steps */
export interface StepDef {
    n: number;
    label: string;
}
export function Steps({ steps, current }: {
    steps: StepDef[];
    current: number;
}) {
    return (<div className="setup-progress"><p className="steps-current small">{tr("Step ")}{tr(current)}{tr(" of ")}{tr(steps.length)} · {tr(steps.find(s => s.n === current)?.label)}</p><div className="steps" role="list" aria-label={tr("Setup progress")}>
      {tr(steps.map((s) => {
            const state = s.n < current ? 'done' : s.n === current ? 'active' : 'todo';
            return (<div key={s.n} className="step" data-state={state} role="listitem" 
            /* Omitted rather than "false" when not current: the attribute is
               supposed to mark one thing, and a query for it should find one. */
            aria-current={state === 'active' ? 'step' : undefined}>
            <span className="step__n">{tr(s.n)}</span>
            <span>·</span>
            <span>{tr(s.label)}</span>
          </div>);
        }))}
    </div></div>);
}
/* ------------------------------------------------------------------ fields */
const FieldContext = createContext<{
    label: string;
    description?: string;
} | null>(null);
export function Field({ label, hint, impact, children, wide, }: {
    label: ReactNode;
    hint?: ReactNode;
    /** What the current value actually costs. Shown where the choice is made
     *  rather than only in a summary elsewhere on the screen. */
    impact?: ReactNode;
    children: ReactNode;
    wide?: boolean;
}) {
    const id = useId();
    return (<FieldContext.Provider value={{ label: `${id}-label`, description: hint || impact ? `${id}-description` : undefined }}>
    <div className="field" role="group" aria-labelledby={`${id}-label`}>
      <div className="field__main">
        <div className="field__label" id={`${id}-label`}>{tr(label)}</div>
        <div id={`${id}-description`}>
          {tr(hint ? <div className="field__hint">{tr(hint)}</div> : null)}
          {tr(impact ? <div className="field__impact">{tr(impact)}</div> : null)}
        </div>
      </div>
      <div className={cx('field__control', wide && 'field__control--wide')}>{tr(children)}</div>
    </div>
    </FieldContext.Provider>);
}
/**
 * A listbox styled as a Strata field.
 *
 * This replaces a native `<select>`, whose popup the OS draws — on macOS a dark
 * system menu with a blue highlight, which shared nothing with the rest of the
 * app. Owning the popup means owning the whole interaction, so this implements
 * the full combobox pattern: focus stays on the trigger with
 * `aria-activedescendant` tracking the highlighted option, and the popup renders
 * through a portal so it can never end up nested inside another glass surface.
 */
export function Select<T extends string>({ value, valueSuffix, options, onChange, ariaLabel, disabled = false, menuSize = 'standard', menuMinWidth, menuWidth, menuClassName, optionIcon, singleLine = false, translateLabels = true, }: {
    value: T;
    valueSuffix?: ReactNode;
    /** Keep custom names and font families verbatim. */
    translateLabels?: boolean;
    /** `title` explains a label that has to stay short to fit the control. */
    options: {
        value: T;
        label: string;
        title?: string;
        disabled?: boolean;
    }[];
    onChange: (v: T) => void;
    ariaLabel?: string;
    disabled?: boolean;
    menuSize?: 'standard' | 'comfortable';
    menuMinWidth?: number;
    /** Override popup width while keeping the trigger unchanged. */
    menuWidth?: number;
    menuClassName?: string;
    optionIcon?: (value: T) => IconName;
    singleLine?: boolean;
}) {
    const field = useContext(FieldContext);
    const listId = useId();
    const triggerRef = useRef<HTMLButtonElement>(null);
    const popupRef = useRef<HTMLDivElement>(null);
    const [open, setOpen] = useState(false);
    const [placed, setPlaced] = useState<{
        left: number;
        top: number;
        width: number;
        up: boolean;
    } | null>(null);
    const [cursor, setCursor] = useState(0);
    const [typed, setTyped] = useState('');
    const current = options.find((o) => o.value === value);
    const optionLabel = (label: string) => translateLabels ? tr(label) : label;
    const close = useCallback((refocus = true) => {
        setOpen(false);
        setTyped('');
        if (refocus)
            triggerRef.current?.focus();
    }, []);
    const openMenu = useCallback(() => {
        const el = triggerRef.current;
        if (!el)
            return;
        const r = el.getBoundingClientRect();
        const height = Math.min(320, options.length * (menuSize === 'comfortable' ? 52 : 38) + 12);
        const below = window.innerHeight - r.bottom;
        // Flip above the trigger when there is more room there than below.
        const up = below < height + 12 && r.top > below;
        const width = Math.min(menuWidth ?? Math.max(r.width, menuMinWidth ?? (menuSize === 'comfortable' ? 340 : 180)), window.innerWidth - 24);
        setPlaced({ left: Math.max(12, Math.min(r.left, window.innerWidth - width - 12)), top: up ? r.top - 6 : r.bottom + 6, width, up });
        setCursor(Math.max(0, options.findIndex((o) => o.value === value)));
        setOpen(true);
    }, [options, value, menuSize, menuMinWidth, menuWidth]);
    // The popup is fixed-positioned, so it has to follow its trigger.
    useEffect(() => {
        if (!open)
            return undefined;
        const reposition = () => {
            const el = triggerRef.current;
            if (!el)
                return;
            const r = el.getBoundingClientRect();
            const height = popupRef.current?.offsetHeight ?? Math.min(320, options.length * (menuSize === 'comfortable' ? 52 : 38) + 12);
            const below = window.innerHeight - r.bottom;
            const up = below < height + 12 && r.top > below;
            const width = Math.min(menuWidth ?? Math.max(r.width, menuMinWidth ?? (menuSize === 'comfortable' ? 340 : 180)), window.innerWidth - 24);
            setPlaced({ left: Math.max(12, Math.min(r.left, window.innerWidth - width - 12)), top: up ? r.top - 6 : r.bottom + 6, width, up });
        };
        const onDocDown = (e: PointerEvent) => {
            if (!triggerRef.current?.contains(e.target as Node) && !popupRef.current?.contains(e.target as Node))
                close(false);
        };
        window.addEventListener('scroll', reposition, true);
        window.addEventListener('resize', reposition);
        document.addEventListener('pointerdown', onDocDown, true);
        return () => {
            window.removeEventListener('scroll', reposition, true);
            window.removeEventListener('resize', reposition);
            document.removeEventListener('pointerdown', onDocDown, true);
        };
    }, [open, options.length, close, menuSize, menuMinWidth, menuWidth]);
    useEffect(() => {
        if (open)
            document.getElementById(`${listId}-${cursor}`)?.scrollIntoView({ block: 'nearest' });
    }, [open, cursor, listId]);
    function onKeyDown(e: React.KeyboardEvent) {
        if (!open) {
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp' || e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openMenu();
            }
            return;
        }
        switch (e.key) {
            case 'Escape':
                e.preventDefault();
                e.stopPropagation();
                close();
                break;
            case 'ArrowDown':
                e.preventDefault();
                setCursor((c) => (c + 1) % options.length);
                break;
            case 'ArrowUp':
                e.preventDefault();
                setCursor((c) => (c - 1 + options.length) % options.length);
                break;
            case 'Home':
                e.preventDefault();
                setCursor(0);
                break;
            case 'End':
                e.preventDefault();
                setCursor(options.length - 1);
                break;
            case 'Enter':
            case ' ':
                e.preventDefault();
                if (!options[cursor].disabled)
                    onChange(options[cursor].value);
                close();
                break;
            case 'Tab':
                close(false);
                break;
            default: {
                // Typeahead: consecutive letters narrow the match, as a native select does.
                if (e.key.length !== 1 || !/\S/.test(e.key))
                    return;
                const next = typed + e.key.toLowerCase();
                const from = next.length === 1 ? cursor + 1 : cursor;
                const ordered = [...options.slice(from), ...options.slice(0, from)];
                const hit = ordered.find((o) => optionLabel(o.label).toLowerCase().startsWith(next));
                const restart = !hit && next.length > 1 ? options.find((o) => optionLabel(o.label).toLowerCase().startsWith(e.key.toLowerCase())) : null;
                const target = hit ?? restart;
                setTyped(restart ? e.key.toLowerCase() : next);
                if (target)
                    setCursor(options.indexOf(target));
            }
        }
    }
    return (<>
      <button ref={triggerRef} type="button" className="st-input select select__trigger" role="combobox" disabled={disabled} aria-expanded={open} aria-haspopup="listbox" aria-controls={open ? listId : undefined} aria-label={tr(ariaLabel)} aria-labelledby={ariaLabel ? undefined : field?.label} aria-describedby={field?.description} aria-activedescendant={open ? `${listId}-${cursor}` : undefined} onClick={() => (open ? close() : openMenu())} onKeyDown={onKeyDown}>
        <span className="select__value">{valueSuffix ? <><span>{optionLabel(current?.label ?? '')}</span>{valueSuffix}</> : optionLabel(current?.label ?? '')}</span>
        <Icon name="chevron" size={16}/>
      </button>
      {tr(open && placed
            ? createPortal(<div ref={popupRef} id={listId} className={cx('select__pop', menuSize === 'comfortable' && 'select__pop--comfortable', singleLine && 'select__pop--single-line', menuClassName)} role="listbox" aria-label={tr(ariaLabel)} data-up={placed.up || undefined} style={{ left: placed.left, top: placed.top, width: placed.width, zIndex: triggerRef.current?.closest('[role="dialog"]') ? 'var(--layer-dialog-popover)' : 'var(--layer-popover)' }}>
              {tr(options.map((o, i) => (<div key={o.value} id={`${listId}-${i}`} role="option" aria-selected={o.value === value} aria-disabled={o.disabled || undefined} tabIndex={-1} className="select__opt" data-active={i === cursor || undefined} onMouseEnter={() => setCursor(i)} onMouseDown={(e) => e.preventDefault()} onClick={() => {
                        if (o.disabled)
                            return;
                        onChange(o.value);
                        close();
                    }}>
                  {tr(optionIcon ? <span className="select__option-icon"><Icon name={optionIcon(o.value)} size={20}/></span> : <span className="select__tick">{tr(o.value === value ? <Icon name="check" size={14}/> : null)}</span>)}
                  <span className="select__option-label" title={o.title ? tr(o.title) : singleLine ? optionLabel(o.label) : undefined}>{optionLabel(o.label)}</span>
                  {tr(optionIcon && o.value === value ? <Icon name="check" size={15} className="select__option-check"/> : null)}
                </div>)))}
            </div>, document.body)
            : null)}
    </>);
}
/** One modal surface for confirmations and narrow-screen inspectors. */
export function Dialog({ open, title, onClose, children, actions, headerActions, compact = false, sheet = false, wide = false, hideTitle = false, className, noClose = false }: {
    open: boolean;
    title: ReactNode;
    onClose: () => void;
    children: ReactNode;
    actions?: ReactNode;
    headerActions?: ReactNode;
    compact?: boolean;
    sheet?: boolean;
    wide?: boolean;
    hideTitle?: boolean;
    className?: string;
    /** No close button: the person must use the dialog's own buttons (a move that is running). */
    noClose?: boolean;
}) {
    const ref = useRef<HTMLElement>(null);
    const id = useId();
    useSurfaceFocus(open, ref);
    useEffect(() => {
        if (!open)
            return;
        const close = (e: KeyboardEvent) => {
            if (e.key === 'Escape' && !e.defaultPrevented && isTopSurface(ref.current)) {
                e.preventDefault();
                e.stopImmediatePropagation();
                onClose();
            }
        };
        document.addEventListener('keydown', close);
        return () => document.removeEventListener('keydown', close);
    }, [open, onClose]);
    if (!open)
        return null;
    return createPortal(<div className="dialog-layer">
    <div className="dialog-scrim" onClick={onClose} aria-hidden="true"/>
    <section ref={ref} className={cx(`${sheet ? 'dialog-surface dialog-surface--sheet' : 'dialog-surface'}${wide ? ' dialog-surface--wide' : ''}${compact ? ' dialog-surface--compact' : ''}`, className)} role="dialog" aria-modal="true" aria-labelledby={id} tabIndex={-1}>
      <header className="dialog-head"><h2 className={hideTitle ? 'sr-only' : 'h3'} id={id}>{tr(title)}</h2><div className="dialog-head__actions">{tr(headerActions)}{noClose ? null : <IconBtn name="x" title={typeof title === 'string' ? tr("Close {0}", [title]) : tr("Close")} onClick={onClose}/>}</div></header>
      <div className="dialog-body">{tr(children)}</div>
      {tr(actions ? <footer className="dialog-actions">{tr(actions)}</footer> : null)}
    </section>
  </div>, document.body);
}
/* ------------------------------------------------------------------- empty */
export function Empty({ icon = 'info', title, body, action, }: {
    icon?: IconName;
    title: string;
    body?: ReactNode;
    action?: ReactNode;
}) {
    return (<div className="empty">
      <span className="empty__icon">
        <Icon name={icon} size={22}/>
      </span>
      <div className="empty__title">{tr(title)}</div>
      {tr(body ? (<div className="small muted" style={{ maxWidth: 460 }}>
          {tr(body)}
        </div>) : null)}
      {tr(action ? <div style={{ marginTop: 6 }}>{tr(action)}</div> : null)}
    </div>);
}
/* ------------------------------------------------------- failed actions ---
   One component for every failure a screen needs to report. The rule this
   encodes (plan 14): wherever a failure is reported, the surface that failed
   offers the retry — a toast may summarise but must not be the only record.
   Kept here so the treatment lands once rather than once per screen. */
export function ActionError() {
    const actionError = useStore((s) => s.actionError);
    if (!actionError)
        return null;
    return <div className="verror row row--between"><ErrorNotice message={`${actionError.label}: ${userError(actionError.message)}`} onRetry={() => actionError.retry()}/><Btn variant="ghost" size="sm" onClick={() => dismissActionError()}>{tr("Dismiss")}</Btn></div>;
}
/* ------------------------------------------------------------------ toasts */
export function Toasts() {
    const toasts = useStore((s) => s.toasts);
    if (!toasts.length)
        return null;
    const iconFor = { success: 'check', info: 'info', warn: 'warning', error: 'error' } as const;
    return (<div className="st-toasts" aria-live="polite">
      {tr(toasts.map((t) => (<div key={t.id} className="st-toast st-toast--toned" data-tone={t.tone} data-leaving={t.leaving || undefined} onMouseEnter={() => holdToast(t.id)} onMouseLeave={() => releaseToast(t.id)} onFocusCapture={() => holdToast(t.id)} onBlurCapture={() => releaseToast(t.id)}>
          <span className="st-toast__icon">
            <Icon name={iconFor[t.tone]} size={20}/>
          </span>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="st-toast__title">{tr(t.title)}</div>
            {tr(t.body ? <div className="st-toast__body">{tr(t.body)}</div> : null)}
          </div>
          <button type="button" className="st-btn st-btn--icon st-btn--sm" onClick={() => dismissToast(t.id)} aria-label={tr("Dismiss")}>
            <Icon name="chevron" size={15} style={{ transform: 'rotate(90deg)' }}/>
          </button>
        </div>)))}
    </div>);
}
/* ------------------------------------------------------------------ metric */
/**
 * A metric tile: the kit's `.st-metric` inside a card, which is how the design
 * reference lays out every number in the app.
 */
export function Metric({ label, value, unit, sub, icon, spark, compact, children, }: {
    label: string;
    value: ReactNode;
    unit?: string;
    sub?: ReactNode;
    icon?: IconName;
    spark?: ReactNode;
    /** Text values read better a step down from the kit's 32px stat size. */
    compact?: boolean;
    children?: ReactNode;
}) {
    return (<section className="st-card">
      <div className="st-metric">
        <div className="st-metric__label">
          {tr(icon ? <Icon name={icon} size={18}/> : null)}
          {tr(label)}
        </div>
        <div className={cx('st-metric__value', compact && 'st-metric__value--text')}>
          {tr(value)}
          {tr(unit ? <span className="st-metric__unit">{tr(unit)}</span> : null)}
        </div>
        {tr(sub ? <div className="spec__sub">{tr(sub)}</div> : null)}
        {tr(children)}
        {tr(spark ? <div className="st-metric__spark">{tr(spark)}</div> : null)}
      </div>
    </section>);
}
/** A row of metric tiles. */
export function MetricGrid({ children, min = 200 }: {
    children: ReactNode;
    min?: number;
}) {
    return (<div className="metricgrid" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))` }}>
      {tr(children)}
    </div>);
}
/** Shared, announced error with a cause and a concrete recovery action. */
export function ErrorNotice({ message, next, action, onRetry, detail, secondary }: {
    message: string;
    next?: string;
    action?: string;
    onRetry?: () => void;
    detail?: string;
    /** A second way out, beside the first (a context error offers two). */
    secondary?: { label: string; onClick: () => void };
}) {
    // A known failure is shown as a sentence and a next step; the original text stays under Details for experts.
    const plain = plainFailure(message);
    const known = plain.kind !== 'plain';
    const nextStep = known ? (plain.next ? tr(plain.next) : next) : next;
    const raw = detail ?? (known ? plain.detail : undefined);
    return <div className="error-notice" role="alert"><Icon name="error" size={18}/><div><strong>{tr(known ? plain.message : userError(message))}</strong>{tr(nextStep ? <p>{tr(nextStep)}</p> : null)}{tr(raw ? <details className="error-detail"><summary className="small">{tr("Details")}</summary><pre className="small muted">{raw}</pre></details> : null)}</div>{tr(onRetry ? (secondary ? <div className="row"><Btn size="sm" onClick={onRetry}>{tr(action ?? 'Try again')}</Btn><Btn size="sm" variant="ghost" onClick={secondary.onClick}>{tr(secondary.label)}</Btn></div> : <Btn size="sm" onClick={onRetry}>{tr(action ?? 'Try again')}</Btn>) : null)}</div>;
}
export function PreviewBadge() { const demo = useStore(s => s.library?.mode === 'simulated'); return demo ? <span className="preview-badge">{tr("Preview build")}</span> : null; }
