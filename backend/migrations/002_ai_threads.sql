CREATE TABLE IF NOT EXISTS ai_threads(
  submission_id TEXT PRIMARY KEY REFERENCES submissions(id),
  draft_json TEXT,
  messages_json TEXT NOT NULL DEFAULT '[]',
  updated REAL NOT NULL
);
