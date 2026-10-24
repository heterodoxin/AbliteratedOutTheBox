import { tr } from '../lib/i18n';
import { GuidedSetup } from '../components/GuidedSetup';
import { useNative } from '../lib/native';
import { useStore } from '../lib/store';
import { useEngineUnpacking } from '../lib/engine-setup';
/** Every setup entry uses the same guided onboarding. */
export function SetupScreen() {
    const native = useNative(), setup = useStore(s => s.setup);
    const unpacking = useEngineUnpacking(!native || !setup);
    if (!native || !setup)
        return <div className="setup guided-onboarding"><p className="onboard-loading" role="status">{tr("Getting your setup ready\u2026")}</p>{unpacking ? <p className="onboard-loading small" role="status">{tr("Unpacking the engine that came with the app. This takes about a minute the first time.")}</p> : null}</div>;
    return <GuidedSetup />;
}
