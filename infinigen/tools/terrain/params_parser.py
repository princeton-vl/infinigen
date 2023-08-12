# Copyright (c) Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Zeyu Ma


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_path', type=str)
args = parser.parse_args()

output = ""
current_type = ""
current_vars = []

code = ""

def get_code(current_type, variables):
    code = ""
    for i, v in enumerate(variables):
        code += f"    {current_type} {v} = {current_type[0]}_params[{i}];\n"
    return code

with open(args.file_path, "r") as f:
    lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("/* params"):
            while True:
                i += 1
                if lines[i].rstrip().endswith(":"):
                    if current_type != "":
                        code += get_code(current_type, current_vars)
                    if lines[i].lstrip().startswith("int"):
                        current_type = "int"
                        current_vars = []
                    elif lines[i].lstrip().startswith("float"):
                        current_type = "float"
                        current_vars = []
                elif lines[i].rstrip().endswith("*/"):
                    code += get_code(current_type, current_vars)
                    break
                else:
                    current_vars.extend([x.lstrip().rstrip() for x in lines[i].lstrip().rstrip().rstrip(',').split(",") if x.lstrip().rstrip() != ""])
        i += 1
                        
print(code)
