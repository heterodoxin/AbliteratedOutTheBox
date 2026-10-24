import { tr } from '../lib/i18n';
import { Icon } from './ui';
/** Native input semantics with a consistent, legible tick across browsers. */
export function OnboardingCheckbox({ checked, onChange }: {
    checked: boolean;
    onChange: (checked: boolean) => void;
}) {
    return <span className="onboard-checkbox"><input type="checkbox" checked={checked} onChange={e => onChange(e.target.checked)}/><span className="onboard-checkbox-mark" aria-hidden="true">{tr(checked ? <Icon name="check" size={13}/> : null)}</span></span>;
}
