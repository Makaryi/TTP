import re

with open('projective/app_web.py', 'r') as f:
    text = f.read()

# Extract the lie detector block from the bottom
block_pattern = r"(try:\n    lie_model_path = os\.path\.join\('models', 'lie_detector_model\.h5'\).*?return jsonify\(\{'error': str\(e\)\}\), 500)"
match = re.search(block_pattern, text, flags=re.DOTALL)
if match:
    lie_block = match.group(1)
    text = text.replace(lie_block, '')
    
    # insert before if __name__ == '__main__'
    text = text.replace("if __name__ == '__main__':", lie_block + "\n\nif __name__ == '__main__':")
    
    # reset port to 8888 because the user liked 8888
    text = text.replace("port=3245", "port=8888")
    text = text.replace("localhost:3245", "localhost:8888")
    
    with open('projective/app_web.py', 'w') as f:
        f.write(text)
    print("Fixed!")
else:
    print("Not found")