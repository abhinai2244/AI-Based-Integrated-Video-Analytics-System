from modules.anpr import is_indian_plate

test_cases = [
    ("MH12AB1234", True),
    ("DL3C1234", True),
    ("KA011234", True),
    ("UP 32 BK 5678", True),
    ("abcd123456", False),
    ("MH12345", False),
    ("MH12345678", False),
    ("12MH1234", False)
]

print("--- Testing Indian Plate Regex ---")
success_count = 0
for text, expected in test_cases:
    result = is_indian_plate(text)
    status = "PASS" if result == expected else "FAIL"
    print(f"Text: '{text}' | Expected: {expected} | Result: {result} | {status}")
    if result == expected: success_count += 1

if success_count == len(test_cases):
    print("\nALL REGEX TESTS PASSED")
else:
    print(f"\n{len(test_cases) - success_count} TESTS FAILED")
