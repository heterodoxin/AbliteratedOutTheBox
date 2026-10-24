import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/kit/tokens.css';
import './styles/kit/components.css';
import './styles/app.css';
import { App } from './app/App';
import { installDropGuard } from './components/DropZone';

const el = document.getElementById('root');
if (!el) throw new Error('#root missing');

installDropGuard();

createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

import './styles/annotation-fixes.css';
import './styles/monitor.css';
import './styles/time-skip.css';

import './styles/model-run.css';

import './styles/onboarding.css';

import './styles/board.css';

import './styles/gpu-roles.css';
