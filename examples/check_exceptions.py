import einx
import types
import numpy as np
import os

report = ""


def check(func_str):
    func_str = func_str.strip()
    markdown = "\nThe following einx code\n"
    markdown += f"```python\n{func_str}\n```\n"
    try:
        exec(func_str)
        markdown += "does not raise an exception even though it should\n"
    except Exception as e:
        markdown += "raises the following exception:\n"
        tb = f"{type(e).__module__}.{type(e).__name__}: {str(e)}"
        markdown += f"```\n{tb}\n```\n"
    markdown += "---\n"
    return markdown


report += "## Syntax Errors\n\n"

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a b -> (a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a b -> a b)", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("!a b -> a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("1a b -> a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("! a b -> a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("(b)(a) -> a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("(b)a -> a b", x)
""")

report += check("""
x = np.zeros((10,))
einx.rearrange("a -> b -> c", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a a -> a b", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.dot("[a b], [a b] -> a", x, x)
""")

report += check("""
x = np.zeros((10,))
einx.rearrange("a, -> (a +)", x, 1)
""")

report += check("""
x = np.zeros((10,))
einx.sum("a [b] c -> a b", x)
""")


report += "\n\n## Dimension Errors\n\n"

report += check("""
x = np.zeros((10, 5))
einx.rearrange("(a b) c -> a b c", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("(a b)... -> a b...", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a b -> b a", x, a=(10, 2))
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("b c d -> d b c", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a... b c d -> b c d a...", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a... b... -> b... a...", x)
""")

report += check("""
x = np.zeros((10,))
einx.rearrange("(a + b) -> a b (1 + 1)", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("(a b) c -> a b c", x, a=7)
""")

report += check("""
x = np.zeros((10, 5))
y = np.zeros((7, 5))
einx.subtract("ba c, i c -> i ba", x, y)
""")


report += "\n\n## Operation-specific Errors\n\n"

report += check("""
x = np.zeros((10, 5))
einx.rearrange("[a] b -> b [a]", x)
""")

report += check("""
x = np.zeros((10, 5))
einx.sum("(a + b) c -> a c", x, a=5)
""")

report += check("""
x = np.zeros((10, 5))
einx.dot("(a + b) c, c (a + b) -> 1", x, x)
""")

report += check("""
x = np.zeros((10, 5))
einx.dot("[a b], [a b] -> [1]", x, x)
""")

report += check("""
x = np.zeros((10, 5))
einx.dot("a [b], a [b] -> 1", x, x)
""")

report += check("""
x = np.zeros((10, 5))
einx.dot("a b, [a b] -> 1", x, x)
""")

report += check("""
x = np.zeros((10, 5))
einx.rearrange("a, b -> a b (1 + 1)", x)
""")

report += check("""
einx.logsumexp("a", [0.0, [1.0]], backend="numpy")
""")

report += check("""
x = np.zeros((10, 5))
y = np.zeros((7,))
einx.add("a b, c", x, y)
""")

# TODO:
# report += check("""
# x = np.zeros((10,))
# einx.vmap("b -> b 3", x, op=lambda x: x + np.zeros((3,)))
# """)

report += check("""
x = np.zeros((10,))
einx.vmap("b -> b 3", x, op=einx.trace(lambda x: x + np.zeros((3,))))
""")

report += check("""
x = np.zeros((4, 5, 6))
y = np.zeros((4, 5), dtype="int32")
einx.get_at("b t [d], b (t [1]) -> b (t [1])", x, y)
""")


file = os.path.join(os.path.dirname(__file__), "check_exceptions")

with open(f"{file}.md", "w") as f:
    f.write(report)

import markdown

html = markdown.markdown(report, extensions=["fenced_code"])
html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: auto; padding: 2em; }}
        code {{ background: #f4f4f4; padding: 2px 4px; border-radius: 4px; }}
        pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; border-radius: 4px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background: #eee; }}
    </style>
</head>
<body>
{html}
</body>
</html>
"""

with open(f"{file}.html", "w") as f:
    f.write(html)

print(report)
