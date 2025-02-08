import zlib
import os

# 读取PCD文件
pcd_file_path = './office1.pcd'
with open(pcd_file_path, 'rb') as f:
    pcd_data = f.read().decode('ascii')

# 解析PCD文件头
header = {}
lines = pcd_data.split('\n')
for line in lines:
    if line.startswith('#') or line.strip() == '':
        continue
    if line.startswith('DATA'):
        header['DATA'] = line.split(' ')[1]
        break
    key, value = line.split(' ', 1)
    header[key] = value

# 检查是否为binary_compressed格式
if header['DATA'] != 'binary_compressed':
    print('文件格式不是binary_compressed，无需转换')
    exit(0)

# 读取二进制数据
with open(pcd_file_path, 'rb') as f:
    binary_data = f.read()
data_start_index = pcd_data.index('DATA binary_compressed') + len('DATA binary_compressed')
compressed_data = binary_data[data_start_index:]

# 解压缩数据
uncompressed_data = zlib.decompress(compressed_data)

# 将二进制数据转换为ASCII格式
point_size = int(header['POINTS']) * int(header['FIELDS']) * 4
points = [float.from_bytes(uncompressed_data[i:i+4], byteorder='little') for i in range(0, len(uncompressed_data), 4)]

# 生成ASCII格式的PCD文件
ascii_content = ''
for line in lines:
    if line.startswith('DATA'):
        ascii_content += 'DATA ascii\n'
        break
    ascii_content += line + '\n'

for i in range(0, len(points), int(header['FIELDS'])):
    ascii_content += ' '.join(map(str, points[i:i+int(header['FIELDS'])])) + '\n'

# 保存转换后的文件
output_file_path = os.path.join(os.path.dirname(pcd_file_path), 'office1_ascii.pcd')
with open(output_file_path, 'w') as f:
    f.write(ascii_content)

print('转换完成，文件已保存为:', output_file_path)
