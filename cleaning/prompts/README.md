# Prompt registry

The production captioning system prompt lives here as a versioned text file.
`registry.json` pins the SHA256 of each version's normalized text.

- Loading is strict: the tool normalizes whitespace (`" ".join(text.split())`),
  hashes the normalized text, and refuses to run when the file hash does not
  match the registry entry. Editing a prompt in place therefore fails loudly.
- To change the prompt, add `caption-v2.txt`, register its normalized SHA256 in
  `registry.json`, and start a new run (the run identity includes the prompt
  version and hash). Do not edit `caption-v1.txt` after a production run starts.
- The prompt file is wrapped for readability only; normalization makes the hash
  insensitive to line wrapping.
- `caption-v2` produces a natural imperative user request rather than a detached
  caption. The image supplies composition while the TikZ source is authoritative
  for visible labels, colors, styles and relationships. Its limit is 300 words.
