import os
import shutil
import patoolib


def decompress_archives(root_dir):
    """智能解压策略：自动判断是否建立外层目录"""
    supported_exts = ('.zip', '.rar', '.7z', '.tar.gz', '.tar', '.gz')

    for foldername, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            ext = os.path.splitext(filename)[1].lower()

            if ext not in supported_exts:
                continue  # 跳过非压缩文件

            # 建立预设的同名子目录
            base_name = os.path.splitext(filename)[0]
            target_dir = os.path.join(foldername, base_name)

            try:
                # 先解压到临时目录检测结构
                temp_dir = os.path.join(foldername, '_temp_')
                os.makedirs(temp_dir, exist_ok=True)

                # 解压到临时目录
                patoolib.extract_archive(file_path, outdir=temp_dir)

                # 检测是否需要提升内容层级
                entries = os.listdir(temp_dir)
                if len(entries) == 1 and os.path.isdir(os.path.join(temp_dir, entries[0])):
                    # 若临时目录只有单个子目录，则直接解压到当前目录
                    shutil.move(os.path.join(temp_dir, entries[0]), foldername)
                    os.rmdir(temp_dir)  # 删除临时目录
                else:
                    # 若临时目录包含多个文件，则移动到正式目标目录
                    shutil.move(temp_dir, target_dir)

                print(f"成功解压 {filename}")
            except Exception as e:
                print(f"解压失败 {filename}: {str(e)}")
            finally:
                # 清理残留临时目录
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    work_dir = r"D:\Music\Pascal\Work"
    decompress_archives(work_dir)