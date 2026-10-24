import { tr } from '../lib/i18n';
/** Settings panels share the existing preference controls. */
import { useRef, type ReactNode } from 'react';
import { Btn, Field, Icon, Select, type IconName } from '../components/ui';
import { Brand } from '../components/Brand';
import { SettingsModels } from '../components/SettingsModels';
import { SettingsTools } from '../components/SettingsTools';
import { SettingsStorage } from '../components/SettingsStorage';
import { PrivacyNotes } from '../components/PrivacyNotes';
import { AboutVersions, SettingsUpdates } from '../components/SettingsUpdates';
import { NativeOperations } from '../components/NativeOperations';
import { FontChoice } from '../components/FontChoice';
import { LANGUAGES } from '../lib/i18n';
import { setLanguage } from '../lib/store';
import { CONTENT_FONT_RANGE, setContentFontSize, setDensity, setFocus, setMotion, setShortcuts, setTheme, setTransparency, setUiScale, UI_SCALES, useStore, setSettingsSection } from '../lib/store';
const PANELS = [{ name: 'Appearance', icon: 'sun', sections: ['General', 'Reading & focus'] }, { name: 'Models', icon: 'models', sections: [] }, { name: 'Tools', icon: 'tool', sections: [] }, { name: 'Storage', icon: 'disk', sections: ['Storage'] }, { name: 'Engine', icon: 'engine', sections: [] }, { name: 'About', icon: 'info', sections: ['About', 'Update', 'Privacy'] }] as const;
export function SettingsScreen() {
    const prefs = useStore(s => s);
    const active = prefs.settingsSection;
    const scrollRef = useRef<HTMLDivElement>(null);
    const fields: {
        section: string;
        label: string;
        hint: string;
        aliases?: string;
        control: ReactNode;
    }[] = [
        { section: 'General', label: 'Language', hint: 'Choose the language used throughout Strata.', control: <Select translateLabels={false} ariaLabel={tr("Language")} value={prefs.language} options={[...LANGUAGES]} onChange={value => setLanguage(value as typeof prefs.language)}/> },
        { section: 'Reading & focus', label: 'Font', hint: 'Choose an installed font for the interface and conversations.', control: <FontChoice /> },
        { section: 'General', label: 'Keyboard shortcuts', hint: 'Set the key combinations that feel natural to you.', aliases: 'command keys hotkeys', control: <Btn size="sm" onClick={() => setShortcuts(true)}>{tr("Edit shortcuts")}</Btn> },
        { section: 'Reading & focus', label: 'Appearance', hint: 'System follows your computer by default.', aliases: 'theme light dark system automatic', control: <Select value={prefs.theme} ariaLabel={tr("Appearance")} options={[{ value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }, { value: 'system', label: 'System' }]} onChange={setTheme}/> },
        { section: 'Reading & focus', label: 'Reading size', hint: 'Conversation text size. Interface controls stay separate.', aliases: 'font text zoom', control: <label className="row"><input aria-label={tr("Reading text size")} type="range" className="st-slider" min={CONTENT_FONT_RANGE.min} max={CONTENT_FONT_RANGE.max} value={prefs.contentFontSize} onChange={e => setContentFontSize(Number(e.target.value))}/><output>{tr(prefs.contentFontSize)}{tr(" px")}</output></label> },
        { section: 'Reading & focus', label: 'Interface size', hint: 'Scale navigation and controls.', aliases: 'zoom scale 90% 100% 110% 125%', control: <Select value={String(prefs.uiScale)} ariaLabel={tr("Interface size")} options={UI_SCALES.map(v => ({ value: String(v), label: `${Math.round(v * 100)}%` }))} onChange={v => setUiScale(Number(v))}/> },
        { section: 'Reading & focus', label: 'Spacing', hint: 'Comfortable spacing by default.', aliases: 'density comfortable compact', control: <Select value={prefs.density} ariaLabel={tr("Interface spacing")} options={[{ value: 'comfortable', label: 'Comfortable' }, { value: 'compact', label: 'Compact' }]} onChange={setDensity}/> },
        { section: 'Reading & focus', label: 'Motion', hint: 'Reduced motion also disables smooth scrolling.', aliases: 'animation system full reduced', control: <Select value={prefs.motion} ariaLabel={tr("Motion")} options={[{ value: 'system', label: 'Follow system' }, { value: 'full', label: 'Full' }, { value: 'reduced', label: 'Reduced' }]} onChange={setMotion}/> },
        { section: 'Reading & focus', label: 'Transparency', hint: 'Translucent controls or solid surfaces. Text stays opaque.', aliases: 'blur glass liquid opaque translucent', control: <Select value={prefs.transparency} ariaLabel={tr("Transparency")} options={[{ value: 'translucent', label: 'Translucent' }, { value: 'opaque', label: 'Opaque' }]} onChange={setTransparency}/> },
        { section: 'Reading & focus', label: 'Focus reading', hint: 'Hides navigation in chat. Settings stays open above the conversation.', aliases: 'distraction quiet', control: <button className="st-toggle" role="switch" aria-checked={prefs.focus} aria-label={tr("Focus reading")} onClick={() => setFocus(!prefs.focus)}/> },
    ];
    function activate(name: typeof PANELS[number]['name']) {
        setSettingsSection(name);
        scrollRef.current?.scrollTo({ top: 0 });
    }
    return <div className="settings-layout">
    <nav className="settingsindex" role="tablist" aria-label={tr("Settings sections")} onKeyDown={e => {
            if (!['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key))
                return;
            e.preventDefault();
            const index = PANELS.findIndex(p => p.name === active);
            const next = e.key === 'Home' ? 0 : e.key === 'End' ? PANELS.length - 1 : (index + (['ArrowDown', 'ArrowRight'].includes(e.key) ? 1 : -1) + PANELS.length) % PANELS.length;
            activate(PANELS[next].name);
            document.getElementById(`settings-tab-${PANELS[next].name}`)?.focus();
        }}>
      {tr(PANELS.map(panel => <button key={panel.name} role="tab" id={`settings-tab-${panel.name}`} aria-controls={`settings-panel-${panel.name}`} aria-selected={active === panel.name} tabIndex={active === panel.name ? 0 : -1} className="settingsindex__item" onClick={() => activate(panel.name)}><Icon name={panel.icon as IconName} size={17}/>{tr(panel.name)}</button>))}
    </nav>
    <div className="settings-pane" ref={scrollRef}>
      {tr(PANELS.map(panel => <div key={panel.name} role="tabpanel" tabIndex={0} id={`settings-panel-${panel.name}`} aria-labelledby={`settings-tab-${panel.name}`} hidden={active !== panel.name}>
        {tr(panel.name === 'Models' ? <SettingsModels /> : panel.name === 'Tools' ? (active === panel.name ? <SettingsTools /> : null) : panel.name === 'Engine' && active === panel.name ? <NativeOperations /> : active === panel.name ? <div className="stack stack--lg settings-body">
          {tr(panel.sections.map(section => <section className="settings-section" key={section}>
            <h2 className="h3" style={{ marginBottom: 14 }}>{tr(section)}</h2>
            <div className="st-card st-card--flush">
              {tr(section === 'About' ? <div className="settings-about-brand"><Brand variant="lockup" height={28}/><AboutVersions/></div> : null)}
              {tr(section === 'Storage' ? <SettingsStorage /> : null)}
              {tr(section === 'Privacy' ? <PrivacyNotes /> : null)}
              {tr(section === 'Update' ? <SettingsUpdates /> : null)}
              {tr(fields.filter(f => f.section === section).map(f => <Field key={f.label} label={tr(f.label)} hint={tr(f.hint)}>{tr(f.control)}</Field>))}
            </div>
          </section>))}
        </div> : null)}
      </div>))}
    </div>
  </div>;
}
