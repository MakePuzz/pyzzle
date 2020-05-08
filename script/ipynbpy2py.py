"""
ipynbファイルから変換したpyファイルを, 指定したパッケージ名のもと, クラス毎に分割する.
コマンドライン引数は
 1. ipynbから得たpyファイル
 2. パッケージ名（-nまたは--nameオプションで指定. デフォルトは'src'）
 3. 追加のipynbから得たpyファイル（-aまたは--addオプションで指定. 複数の場合は-aオプションを複数回つけて指定）
 4. __pycache__ディレクトリをgitignoreするかどうか（-niまたは--notignoreと書くとignoreをしない. デフォルトはignoreする.）

 実行例：
 python ipynbpy2py.py jupyter/CrosswordLocalSearch.py -n sample_package
 python ipynbpy2py.py jupyter/CrosswordLocalSearch.py -n sample_package -a jupyter/CrosswordExtension.py
 python ipynbpy2py.py jupyter/CrosswordLocalSearch.py -n sample_package -a jupyter/CrosswordExtension.py -a jupyter/CrosswordExtension_2.py
"""

import os
import argparse
from pathlib import Path

# In[]
parser = argparse.ArgumentParser(description="convert ipynb.py to .py with given package_name")
parser.add_argument("ipynbpy", type=str,
                    help="python file made by jupytext or ipynb")
parser.add_argument("-a", "--add", type=str, action='append',
                    help="additional python file made by jupytext or ipynb")
parser.add_argument("-n", "--name", type=str, default="src",
                    help="name of package, default=src")
parser.add_argument("-ni", "--notignore", action='store_true',
                    help="do not make a .gitignore file for __pycache__, default=False")
args = parser.parse_args()

# settings
ipynbpy = args.ipynbpy
package_path = args.name
additional_ipynbpy = args.add
not_ignore = args.notignore

package_name = Path(package_path).name

# open
with open(ipynbpy, encoding='utf-8') as f:
    lines = f.readlines()
if additional_ipynbpy != None:
    for add_f in additional_ipynbpy:
        with open(add_f, encoding='utf-8') as f:
            add_lines = f.readlines()
        for add_line in add_lines:
            lines.append(add_line)

# read import and class
imports_all, imports, classes = [], [], []
for i, line in enumerate(lines):
    if line[:7] == "import " or line[:5] == "from ":
        imports_all.append(line)
    if line[:6] == "class ":
        classes.append(line)
imports_all = list(set(imports_all))
class_names = list(map(lambda c: c[6:-2], classes))

# remove unused imports
for import_line in imports_all:
    for class_name in class_names:
        if class_name not in import_line and import_line not in imports:
            imports.append(import_line)

# set class line box, if 5 class is loaded, class_lines = [[],[],[],[],[]]
import_table, import_lines, class_lines = [], [], []
for _ in range(len(classes)):
    import_table.append([])
    import_lines.append([])
    class_lines.append([])
# get class lines written by class Hoge(): area
class_flag = False
class_num = -1
for line in lines:
    if line[:6] == "class ":
        class_flag = True
        class_num += 1
    elif line[0] not in (" ", "\n"):
        class_flag = False
    if class_flag is True:
        class_lines[class_num].append(line)
# get class lines written by def fuga ... setattr(Hoge, "fuga", fuga)
def_flag = False
def_end_flag = False
for line in lines:
    if line[:4] == "def ":
        def_lines = []
        def_flag = True
    elif line[0] not in (" ", "\n", "setattr"):
        def_flag = False
    if def_flag is True:
        def_lines.append(line)
    if line[:7] == "setattr":
        def_end_flag = True
        setattr_class = line.split(",")[0][8:]
    if def_end_flag is True:
        class_num = class_names.index(setattr_class)
        for def_line in def_lines:
            if def_line[0] not in ("\n"):
                def_line = "    " + def_line
            class_lines[class_num].append(def_line)
        def_end_flag = False

# 'import' arrangement
import_names = []
for import_line in imports:
    if "as" in import_line:
        import_names.append(import_line.split("as ")[-1].rstrip().lstrip())
    else:
        name = import_line.split("import ")[-1]
        if len(name.split(",")) is 1:
            import_names.append(name.rstrip().lstrip())
        else:
            for name in name.split(","):
                import_names.append(name.rstrip().lstrip())
for class_num, class_line in enumerate(class_lines):
    for line in class_line:
        for import_name in import_names:
            if ","+import_name+"." in line or ","+import_name+"(" in line or " "+import_name+"." in line or " "+import_name+"(" in line:
                import_table[class_num].append(import_name)
        for class_name in class_names:
            if ","+class_name+"." in line or ","+class_name+"(" in line or " "+class_name+"." in line or " "+class_name+"(" in line:
                import_table[class_num].append(class_name)
    import_table[class_num] = list(set(import_table[class_num]))
for class_num, class_name in enumerate(class_names):
    if class_name in import_table[class_num]:
        import_table[class_num].remove(class_name)
for class_num in range(len(class_names)):
    for import_name in import_table[class_num]:
        for import_line in imports:
            if import_name in import_line:
                import_lines[class_num].append(import_line)
    import_lines[class_num] = list(set(import_lines[class_num]))
    import_lines[class_num].append("\n")
for class_num in range(len(class_names)):
    for import_name in import_table[class_num]:
        for class_name in class_names:
            if import_name in class_name:
                import_line = f"from {package_name}.{class_name} import {class_name}\n"
                import_lines[class_num].append(import_line)
    import_lines[class_num].append("\n")

## output
os.makedirs(package_path, exist_ok=True)
# __init__.py
with open(f'{package_path}/__init__.py', 'w', encoding='utf-8') as of:
    for class_name in class_names:
        of.write(f"from {package_name}.{class_name} import {class_name}\n")
# class_name.py
for class_num, class_name in enumerate(class_names):
    with open(f'{package_path}/{class_name}.py', 'w', encoding='utf-8') as of:
        for import_line in import_lines[class_num]:
            of.write(import_line)
        for class_line in class_lines[class_num]:
            of.write(class_line)
# .gitignore
if not_ignore is False:
    with open(f'{package_path}/.gitignore', 'w', encoding='utf-8') as of:
        of.write(f"__pycache__/\n")
