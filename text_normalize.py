"""Normalize technical/code text into speakable words for TTS."""

import re


def normalize_for_speech(text):
    """Convert code/technical notation into speakable words."""

    # Strip code blocks and inline code that slipped through
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # URLs and domains: remove FIRST before dots get rewritten
    text = re.sub(r'https?://[^\s,)]+', '', text)
    text = re.sub(r'\b[\w.-]+\.(com|org|net|io|dev|ai)\b[^\s,)]*', '', text)

    # Common abbreviations: protect from the file-extension rule
    text = re.sub(r'\be\.g\.', 'for example', text)
    text = re.sub(r'\bi\.e\.', 'that is', text)
    text = re.sub(r'\betc\.', 'etcetera', text)
    text = re.sub(r'\bvs\.', 'versus', text)

    # File paths: /Users/foo/bar -> just the last component
    text = re.sub(r'(?:/[\w._-]+){2,}', lambda m: m.group().rsplit('/', 1)[-1], text)

    # File extensions: say.py -> say dot py, config.json -> config dot json
    text = re.sub(r'(\w)\.([a-zA-Z]{1,4})\b', r'\1 dot \2', text)

    # Flags: --queue -> dash dash queue, -v -> dash v
    text = re.sub(r'--([a-zA-Z][\w-]*)', r'dash dash \1', text)
    text = re.sub(r'\s-([a-zA-Z])\b', r' dash \1', text)

    # Hex/binary literals: 0xFF -> hex FF, 0b1010 -> binary 1010
    text = re.sub(r'\b0[xX]([0-9a-fA-F]+)\b', r'hex \1', text)
    text = re.sub(r'\b0[bB]([01]+)\b', r'binary \1', text)

    # CamelCase: backgroundColor -> background Color
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Special names: C++ -> C plus plus, C# -> C sharp
    text = re.sub(r'\bC\+\+', 'C plus plus', text)
    text = re.sub(r'\bC#', 'C sharp', text)

    # Shell/env variables: $VARIABLE -> variable
    text = re.sub(r'\$\{?([a-zA-Z_]\w*)\}?', r'\1', text)

    # Asterisks: **kwargs -> kwargs, *args -> args
    text = re.sub(r'\*{1,2}(\w+)', r'\1', text)

    # Shell operators
    text = text.replace('&&', ' and ')
    text = text.replace('||', ' or ')
    text = re.sub(r'\s*\|\s*', ' pipe ', text)

    # Common symbols (order matters — multi-char before single-char)
    text = text.replace('...', ', ')
    text = text.replace('==', ' equals ')
    text = text.replace('!=', ' not equals ')
    text = text.replace('>=', ' greater than or equal to ')
    text = text.replace('<=', ' less than or equal to ')
    text = text.replace('=>', ' arrow ')
    text = text.replace('->', ' arrow ')
    text = re.sub(r'~/', 'home directory slash ', text)
    text = text.replace('/', ' slash ')
    text = text.replace('_', ' ')
    text = text.replace('~', ' ')
    text = re.sub(r'#(\d+)', r'number \1', text)  # issue #123
    text = text.replace('#', '')
    text = re.sub(r'@(\w+)', r'\1', text)  # @decorator -> decorator
    text = re.sub(r'(\d+)%', r'\1 percent', text)

    # Brackets and braces: just remove
    text = re.sub(r'[{}\[\]<>()]+', ' ', text)

    # Collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
