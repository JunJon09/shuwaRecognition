import os


#LSA64のデータセットをトレーニング用とテスト用に分割
def DatasetSplit():
    folder_path = "/Users/jonmac/jon/研究/手話/手話単語判別/shuwa/data/all/"
    files_and_directories = os.listdir(folder_path)

# ファイル名のみを取得（ディレクトリを除外）
    file_names = [f for f in files_and_directories if os.path.isfile(os.path.join(folder_path, f))]
    sorted_file_names = sorted(file_names)
    count = 1
    train_list = []
    test_list = []
    for file_name in sorted_file_names:
        if count <= 40:
            train_list.append(file_name)
        elif 41 <= count and count <= 49:
            test_list.append(file_name)
        elif count == 50:
            test_list.append(file_name)
            count = 0
        count += 1

    return train_list, test_list

