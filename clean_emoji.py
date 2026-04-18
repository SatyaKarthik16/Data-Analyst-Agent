import re
from pathlib import Path

def remove_emojis(text):
    # Regex to match emojis (basic range)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002700-\U000027BF"  # dingbats
        "\U0001f926-\U0001f937"  # gestures
        "\U00010000-\U0010ffff"  # other unicode
        "\u2640-\u2642"  # gender symbols
        "\u2600-\u2B55"  # misc symbols
        "\u200d"  # zero width joiner
        "\u23cf"  # eject symbol
        "\u23e9"  # fast forward
        "\u231a"  # watch
        "\ufe0f"  # variation selector
        "\u3030"  # wavy dash
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

for path in Path('.').rglob('*'):
    if path.suffix in {'.py', '.md'} and path.name != 'clean_emoji.py':  # Exclude self
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
            cleaned = remove_emojis(text)
            if cleaned != text:
                path.write_text(cleaned, encoding='utf-8')
                print(f'Updated {path}')
        except Exception as e:
            print(f'Error processing {path}: {e}')
