import type { IconName } from '../components/ui';

export function modelFamilyIcon(family?: string): IconName {
  return /^qwen/i.test(family ?? '') ? 'family-qwen' : 'models';
}
