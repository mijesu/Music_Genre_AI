import os
import datetime
import ctypes
import locale


def is_hidden(filepath):
    """检查 Windows 隐藏文件属性"""
    try:
        attrs = ctypes.windll.kernel32.GetFileAttributesW(filepath)
        return attrs & 0x2  # FILE_ATTRIBUTE_HIDDEN
    except:
        return False


def detect_encoding(content_bytes):
    """检测文件编码是否为 GB2312 或系统 ANSI 编码"""
    try:
        content_bytes.decode('gb2312')
        return 'gb2312'
    except UnicodeDecodeError:
        pass

    try:
        ansi_encoding = locale.getpreferredencoding()
        content_bytes.decode(ansi_encoding)
        return ansi_encoding
    except UnicodeDecodeError:
        return None


def convert_cue_files(source_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith('.cue'):
                continue

            file_path = os.path.join(root, file)
            if is_hidden(file_path):
                continue

            try:
                # 读取二进制内容
                with open(file_path, 'rb') as f:
                    content_bytes = f.read()

                # 检测编码
                encoding = detect_encoding(content_bytes)
                if not encoding:
                    error_msg = f"檔案 {file} 不是 GB2312 或 ANSI 編碼，未進行轉換。\n時間: {datetime.datetime.now()}"
                    error_path = os.path.join(root, 'ERROR.TXT')
                    with open(error_path, 'w', encoding='utf-8-sig') as f_error:
                        f_error.write(error_msg)
                    print(f"[警告] 非 GB2312/ANSI 編碼: {file_path}")
                    continue

                # 生成新文件名
                new_name = f"{os.path.splitext(file)[0]}UTF8{os.path.splitext(file)[1]}"
                dest_path = os.path.join(root, new_name)

                # 转换并写入 UTF-8 with BOM
                content = content_bytes.decode(encoding)
                with open(dest_path, 'w', encoding='utf-8-sig') as f_out:
                    f_out.write(content)
                print(f"[成功] 轉換完成: {file_path} => {dest_path}")

            except Exception as e:
                error_msg = f"處理檔案 {file} 時發生錯誤: {str(e)}\n時間: {datetime.datetime.now()}"
                error_path = os.path.join(root, 'ERROR.TXT')
                with open(error_path, 'w', encoding='utf-8-sig') as f_error:
                    f_error.write(error_msg)
                print(f"[錯誤] 轉換失敗: {file_path} ({str(e)})")


if __name__ == "__main__":
    source_directory = r"D:\Music\Pascal\Work"
    convert_cue_files(source_directory)
    print("\n全部文件已處理！新文件保存至原目錄結構中，並添加 UTF8 後綴。")