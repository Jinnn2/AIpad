from pathlib import Path

path = Path('lineart-board/app/graph_runtime.py')
text = path.read_text('utf-8', errors='ignore')
old = (
    '                    "浣犳鍦ㄧ淮鎶ょ敾甯冪煡璇嗗潡銆俓n"\n'
    '                    "1. �?20瀛椾互鍐呴噸鍐欒鍧楃殑鎽樿锛岃鐩栧綋鍓嶅叏閮ㄨ鐐广€俓n"\n'
    '                    "2. 鎺ㄦ柇璇ュ潡涓庡叾瀹冨潡鐨勮�?鍔熻�?瑙嗚鍏崇郴銆俓n"\n'
    '                    "浠呰繑鍥?JSON {\"summary\": str, \"relationships\": [{\"type\": str, \"target\": str, \"score\": float? ...}]}銆俓n"\n'
)
new = (
    '                    "You are maintaining the structured knowledge blocks on this canvas.\\n"\n'
    '                    "1. Rewrite the block summary so it covers all current fragments (aim for 120 characters or fewer).\\n"\n'
    '                    "2. Identify relationships between this block and other blocks (semantic, functional, or visual flow).\\n"\n'
    '                    "Return JSON {\\"summary\\": str, \\\"relationships\\": [{\\"type\\": str, \\"target\\": str, \\"score\\": float? ...}]}. Use relationship types such as refines, comment_on, or flow_next. Skip any relationship you cannot justify.\\n"\n'
)
if old not in text:
    raise SystemExit('original prompt snippet not found')
text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
