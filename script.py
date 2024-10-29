import subprocess

# 定义输入和输出文件夹路径
input_folder = './outputs/bookcases/'
output_folder = './outputs/bookcases/obj/'
file_format = 'obj'
resolution = 1024

factories = ['SimpleBookcaseFactory', 'BookStackFactory', 'NatureShelfTrinketsFactory']
num_of_class = [1, 2, 6]

for i in range(3):
    for j in range(num_of_class[i]):
        command = [
            'python',
            '-m',
            'infinigen.tools.export',
            '--input_folder', input_folder+factories[i]+'_00'+str(j)+'/',
            '--output_folder', output_folder+factories[i]+str(j)+'/',
            '-f', file_format,
            '-r', str(resolution)
        ]
        try:
            subprocess.run(command, check=True)
            print("命令执行成功！")
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败：{e}")

