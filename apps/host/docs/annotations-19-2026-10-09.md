# Browser review: 19 annotations

Implemented in the restored app at `http://localhost:5160`.

| Annotation | Change |
| --- | --- |
| 1 | A compact 440px task delete dialog explains that the schedule stops and the instructions/last result remain recoverable. The row collapses over 240ms and an animated success toast confirms deletion. Reduced motion is respected. |
| 2–4 | The connection guide is titled “Connect Strata to an app.” OpenAI/Anthropic tabs are in its header immediately before Close. The example-template sentence is removed. |
| 5 | API key status and controls are part of the connection summary alongside Model ID, Status and Local network. |
| 6, 8, 9, 11, 13, 15 | Screen headers no longer display Preview build badges. |
| 7 | Serving's subtitle is removed. |
| 10, 12 | Model's Engine button and Engine's Model version button are removed from their headers. |
| 14 | Monitor's current model appears after its title on the same line. |
| 16–17 | Skills header actions are Add skill, Search, Refresh. The section heading is “Manage your skills library.” |
| 18 | Chat's model badge is a version menu showing name, quantization, size and availability. It supports downloads and switching through the existing host operations, including their lifecycle gates. A switch applies to all chats and retains conversations. |
| 19 | Browser speech recognition writes interim/final text into the existing draft without submitting. Dictation has a right-side feedback panel, pause detection, Stop/Done controls and a configurable Cmd/Ctrl+Shift+D shortcut. Escape closes the panel and releases capture. Navigation, manual draft edits and response generation stop capture. |

## Verification

- Typecheck and production build passed.
- Twelve focused tests passed: six dictation lifecycle tests, three agent update tests, and three task/sharing tests.
- Browser checks covered all requested headers and control positions, skill search filtering, both protocol templates, the compact deletion dialog, deletion feedback, and the customizable dictation shortcut.
- A paused “Deletion feedback check” task was created and archived through the UI. Its instructions remain in Deleted tasks; it never ran.
- The simulated model was downloaded and switched to Smallest, then restored to the original Balanced version. No real model files were downloaded.
- Screenshots are in `docs/evidence/annotations-19-2026-10-09/`.

## Dictation support

Recognition uses the browser's `SpeechRecognition` or `webkitSpeechRecognition` service. Microphone permission and a working browser speech service are required; some services process audio online. Missing support, denied permission, disconnected microphones, service/network errors and idle capture have explicit handling. The app does not provide its own speech backend.

The 3.5-second pause timer, interim corrections, final results, errors and cleanup were tested with simulated recognition events. Live microphone transcription was not exercised. Browser services may end recognition independently of the app's timer.

References: [SpeechRecognition](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition), [interim results](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/interimResults), [stop](https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition/stop).
