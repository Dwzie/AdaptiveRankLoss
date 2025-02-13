# 计算两个fasta文件的序列相似度
def extract_names_and_count_symbols(file_path):
    names = set()
    symbol_count = 0
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>sp|'):
                name = line.split('|')[1]
                names.add(name)
            symbol_count += line.count('>')
    return names, symbol_count


def compare_files(file1_path, file2_path):
    names1, symbol_count1 = extract_names_and_count_symbols(file1_path)
    names2, symbol_count2 = extract_names_and_count_symbols(file2_path)

    intersection = names1.intersection(names2)
    union = names1.union(names2)

    overlap_percentage = len(intersection) / len(union) * 100

    unique_to_file1 = names1 - names2
    unique_to_file2 = names2 - names1

    return overlap_percentage, symbol_count1, symbol_count2, unique_to_file1, unique_to_file2


def write_names_to_file(file_path, names):
    with open(file_path, 'w') as file:
        for name in names:
            file.write(f'>{name}\n')



# file1_path = 'uniprotkb_human_AND_reviewed_true_2024_04_15.fasta'
file1_path = 'uniprotkb_9606_AND_reviewed_true_AND_mo_2024_04_22.fasta'
file2_path = 'uniprotkb_9606_AND_reviewed_true_AND_mo_2024_04_22iso.fasta'
overlap_percentage, symbol_count1, symbol_count2, unique_to_file1, unique_to_file2 = compare_files(file1_path,
                                                                                                   file2_path)
print(f'重合度：{overlap_percentage:.2f}%')
print(f'文件1中"<"符号的个数：{symbol_count1}')
print(f'文件2中"<"符号的个数：{symbol_count2}')
print('文件1中独有的名称:')

write_names_to_file('unique_to_file1.txt', unique_to_file1)
write_names_to_file('unique_to_file2.txt', unique_to_file2)
