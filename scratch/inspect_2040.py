import json

transcript_path = r'C:\Users\harsh\.gemini\antigravity\brain\711df14d-c885-4739-99cf-7003c19e3fa8\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 2040 <= i <= 2060:
            obj = json.loads(line)
            src = str(obj.get('source', ''))
            content = str(obj.get('content', '')).encode('ascii', 'ignore').decode('ascii')
            calls = obj.get('tool_calls', [])
            print(f"=== Line {i} {src} ===")
            if calls:
                for c in calls:
                    print("CALL:", c.get('name'), c.get('arguments'))
            if content.strip():
                print(content[:300])
