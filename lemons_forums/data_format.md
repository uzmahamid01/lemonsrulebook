# Forum Tree Output Format

One file containing a sequence of JSON objects:

- `type`: "board", "thread", or "post"
- `title`: The title/subject
- `content`: The actual content/message  
- `author`: Who wrote it
- `creation_time`: When it was created (ISO format)
- `path`: The hierarchical path (forums → threads → posts)
- `url`: The original URL

Empty fields are omitted.

## Example

```json
{"type":"board","path":["14"],"url":"https://forums.example.com/viewforum.php?id=14","title":"General Discussion"}
{"type":"thread","path":["14","1234"],"url":"https://forums.example.com/viewtopic.php?id=1234","title":"Welcome to the forum","author":"admin","creation_time":"2024-01-15T10:30:00Z"}
{"type":"post","path":["14","1234","5678"],"url":"https://forums.example.com/viewtopic.php?pid=5678","content":"Thanks for joining us!","author":"user123","creation_time":"2024-01-15T11:45:00Z"}
```

## Python Examples

See `search_example.py` for a simple forum search example:

```bash
python3 search_example.py
```

The example shows how to find thread IDs containing specific keywords in post content.



fails a specific Rule:
Rule fails because they fail to understand.
what are the things we can predict people can get wrong. 