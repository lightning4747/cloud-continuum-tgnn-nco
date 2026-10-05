import re

with open("paper/main.tex", "r") as f:
    content = f.read()

# Check key sections
sections = re.findall(r'\\section\{([^}]+)\}', content)
print("Sections found:", sections)
subsections = re.findall(r'\\subsection\{([^}]+)\}', content)
print("Subsections found:", len(subsections), subsections[:5])
