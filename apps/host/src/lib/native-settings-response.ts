/** The UI must never confuse an HTTP error body with a model configuration. */
export function nativeSettingsValues(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Native settings must be an object.');
  }
  return value as Record<string, unknown>;
}
export function nativeSettingsResponse(input: unknown) {
  const response = nativeSettingsValues(input);
  if (typeof response.error === 'string') throw new Error(response.error);
  return {
    values: nativeSettingsValues(response.values),
    mcp: response.mcp === undefined ? {} : nativeSettingsValues(response.mcp),
  };
}
