import json

transcript_path = r'C:\Users\harsh\.gemini\antigravity\brain\711df14d-c885-4739-99cf-7003c19e3fa8\.system_generated\logs\transcript.jsonl'
with open(transcript_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 1950 <= i <= 2075:
            obj = json.loads(line)
            src = str(obj.get('source', ''))
            content = str(obj.get('content', '')).encode('ascii', 'ignore').decode('ascii')
            calls = obj.get('tool_calls', [])
            call_info = ''
            if calls:
                for c in calls:
                    call_info += f" [Tool: {c.get('name')}]"
            if 'output' in content or 'matching' in content or 'submit' in content.lower():
                print(f"Line {i} {src}{call_info}: {content[:160]}")
