/** A small note model for the Fieldnotes prototype. */
export interface Note {
  id: string;
  title: string;
  body: string;
  updatedAt: string;
  pinned: boolean;
}

export function createNote(title = 'A new idea'): Note {
  return {
    id: crypto.randomUUID(),
    title,
    body: '',
    updatedAt: new Date().toISOString(),
    pinned: false,
  };
}

export function recentNotes(notes: Note[]): Note[] {
  return [...notes].sort((a, b) =>
    Number(b.pinned) - Number(a.pinned) ||
    b.updatedAt.localeCompare(a.updatedAt)
  );
}
