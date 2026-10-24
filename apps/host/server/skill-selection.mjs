/** Match enabled instructions within their saved workspace scope. */
export function selectSkills(skills, message, workspaceId) {
  const text = String(message ?? '').toLowerCase();
  return skills.filter(skill => skill.enabled && (!skill.onlyIn || skill.onlyIn === workspaceId)
    && typeof skill.body === 'string' && skill.body.trim()
    && skill.triggers?.some(trigger => trigger && text.includes(String(trigger).toLowerCase())));
}

export function skillInstructions(skills) {
  return skills.map(skill => `Skill: ${skill.name}\n${skill.body.trim()}`).join('\n\n');
}
