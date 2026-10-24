import { tr } from '../lib/i18n';

const NOTES = [
  'Strata sends no usage telemetry.',
  'Conversations stay in browser storage.',
  'When you choose an API model, that conversation and enabled skill instructions are sent to its provider.',
  'The UI request log keeps client, timing and token metadata in memory.',
  'Native API monitoring can retain request content in memory when enabled.',
  'Connected tools access the resources you configure.',
];

/** What Strata collects and where things go: plain lines of text, nothing else. */
export function PrivacyNotes() {
  return <div className="privacy-notes" aria-label={tr("Collection and logs")}>
    {NOTES.map(text => <p key={text}>{tr(text)}</p>)}
  </div>;
}
