import os
import ast
import sys
import pkgutil
import importlib.util

project_dir = "."  # Adjust if needed

# List of stdlib modules to ignore
stdlib_modules = set(m.name for m in pkgutil.iter_modules() if importlib.util.find_spec(m.name) and m.module_finder.path is None)

imports = set()

for root, _, files in os.walk(project_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r", encoding="utf-8") as f:
                try:
                    tree = ast.parse(f.read(), filename=path)
                except Exception as e:
                    print(f"Skipping {file} due to parse error: {e}", file=sys.stderr)
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split('.')[0])

# Filter out standard library modules
used_packages = sorted(imports - stdlib_modules)

print("\n".join(used_packages))
