# Rigel

## Developer Notes

### Wikipedia Parser

- The `jsonl` must be opened in binary mode.
  The start and end indices in the index file are calculated in bytes.
  For a file opened in text mode, `seek()` operates in terms of UTF8 codepoints. So the byte indices will yield incorrect results.
  Instead, open the file in binary mode, read the slice of bytes and call `decode()` on it.
