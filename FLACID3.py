import os
import logging
from pathlib import Path
from mutagen.id3 import ID3, TPE2, TALB, TYER, TPE1, TIT2, APIC, TCOM, COMM, TPOS
import mutagen
from mutagen.wave import WAVE
import sys

# 設定標準輸出流的編碼為 UTF-8（適用於繁體中文環境）
sys.stdout.reconfigure(encoding='utf-8')  # Python 3.7+
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# 配置日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tagging.log', encoding='utf-8'),  # 設定日誌文件為 UTF-8
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_directory_structure(work_dir):
    """處理目錄結構與標籤寫入的主函數"""
    base_path = Path(work_dir)

    # 遍歷第一層目錄 (專輯藝術家)
    for album_artist_dir in base_path.iterdir():
        if not album_artist_dir.is_dir():
            continue

        album_artist = album_artist_dir.name
        logger.info(f"處理專輯藝術家: {album_artist}")

        # 遍歷第二層目錄 (藝人 - 年份 - 專輯)
        for sub_dir in album_artist_dir.iterdir():
            if not sub_dir.is_dir():
                continue

            try:
                process_sub_directory(sub_dir, album_artist)
            except Exception as e:
                logger.error(f"處理子目錄失敗: {sub_dir} - {str(e)}")


def process_sub_directory(sub_dir, album_artist):
    """處理單個子目錄"""
    dir_parts = sub_dir.name.split(' - ')
    if len(dir_parts) != 3:
        logger.warning(f"無效目錄格式: {sub_dir.name}")
        return

    artist, year, album = dir_parts
    logger.info(f"解析目錄資訊: 藝人={artist}, 年份={year}, 專輯={album}")

    # 查找封面文件
    cover_path = find_cover_image(sub_dir)

    # 處理音訊文件
    for audio_file in sub_dir.glob('*.*'):
        if audio_file.suffix.lower() not in ['.mp3', '.flac']:
            continue

        process_audio_file(
            audio_file=audio_file,
            album_artist=album_artist,
            artist=artist,
            year=year,
            album=album,
            cover_path=cover_path
        )


def find_cover_image(directory):
    """查找封面圖片"""
    for ext in ['.jpg', '.jpeg', '.png']:
        for name in ['Cover', '封面']:
            path = directory / f"{name}{ext}"
            if path.exists():
                logger.info(f"找到封面圖片: {path}")
                return path
    return None


def process_audio_file(audio_file, album_artist, artist, year, album, cover_path):
    """處理單個音訊文件"""
    try:
        # 解析文件名取得標題
        title = parse_title_from_filename(audio_file.stem, artist)
        logger.info(f"處理文件: {audio_file.name} | 標題={title}")

        # 初始化 ID3 標籤
        audio = init_id3_tag(audio_file)

        # 設定基本標籤
        set_basic_tags(audio, album_artist, artist, year, album, title)

        # 添加專輯封面
        if cover_path:
            add_album_cover(audio, cover_path)

        # 清空不需要的欄位
        clear_unused_fields(audio)

        # 儲存修改
        save_id3_changes(audio, audio_file)

    except Exception as e:
        logger.error(f"文件處理失敗: {audio_file} - {str(e)}")


def parse_title_from_filename(filename, artist):
    """從文件名解析標題"""
    if ' - ' not in filename:
        raise ValueError("無效文件名格式")
    return filename.split(' - ', 1)[1].strip()


def init_id3_tag(audio_file):
    """初始化 ID3 標籤"""
    try:
        return ID3(audio_file)
    except mutagen.id3.ID3NoHeaderError:
        return ID3()


def set_basic_tags(audio, album_artist, artist, year, album, title):
    """設定基本標籤資訊"""
    audio.add(TPE2(encoding=3, text=album_artist))  # 專輯藝術家
    audio.add(TPE1(encoding=3, text=artist))  # 藝人
    audio.add(TALB(encoding=3, text=album))  # 專輯
    audio.add(TYER(encoding=3, text=year))  # 年份
    audio.add(TIT2(encoding=3, text=title))  # 標題


def add_album_cover(audio, cover_path):
    """添加專輯封面"""
    with open(cover_path, 'rb') as f:
        cover_data = f.read()

    mime_type = 'image/jpeg' if cover_path.suffix.lower() in ['.jpg', '.jpeg'] else 'image/png'

    audio.add(APIC(
        encoding=3,
        mime=mime_type,
        type=3,  # 3 表示封面圖片
        desc='Cover',
        data=cover_data
    ))


def clear_unused_fields(audio):
    """清空不需要的欄位"""
    audio.add(TCOM(encoding=3, text=''))  # 作曲者
    audio.add(COMM(encoding=3, text=''))  # 備註
    audio.add(TPOS(encoding=3, text=''))  # 唱片編號


def save_id3_changes(audio, audio_file):
    """儲存 ID3 標籤變更"""
    audio.save(audio_file, v2_version=3)  # 強制儲存為 ID3v2.3 格式
    logger.info(f"成功更新標籤: {audio_file}")


if __name__ == '__main__':
    work_directory = r"D:\Music\Pascal\Work"

    if not os.path.exists(work_directory):
        logger.error(f"工作目錄不存在: {work_directory}")
    else:
        logger.info(f"開始處理工作目錄: {work_directory}")
        process_directory_structure(work_directory)
        logger.info("處理完成")
