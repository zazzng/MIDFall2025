#!/usr/bin/env python3
"""
각 책의 모든 캐릭터가 함께 같은 속도로 제목을 외치도록 TTS를 생성합니다.
제목 뒤에 느낌표를 붙여서 읽습니다.
"""

import os
import json
import subprocess
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = "gpt-4o-mini-tts"   # 음성 생성 모델

# ROLE_MAP (각 책의 캐릭터 - cha1과 cha2가 함께 제목을 외침)
ROLE_MAP = {
    "SCJ": {"cha1": "simcheong",    "cha2": "simbongsa"},
    "HBJ": {"cha1": "heungbu",      "cha2": "nolbu"},
    "BJBJ": {"cha1": "turtle",      "cha2": "rabbit"},
    "OGJJ": {"cha1": "onggojip",    "cha2": "onggojip"},
    "JWCJ": {"cha1": "jeonwoochi",  "cha2": "jeonwoochi"},
    "JHHRJ": {"cha1": "sister_older",    "cha2": "sister_younger"},  # 자매 둘이 함께
    "PSJ": {"cha1": "ugly",         "cha2": "pretty"},
    "DGJ": {"cha1": "toad",         "cha2": "fox"},
    "KWJ": {"cha1": "kimwon",       "cha2": "monster"}
}

def load_characters():
    """characters_tone.json에서 캐릭터 정보를 로드합니다."""
    with open('characters_tone.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_book_title(book_code: str) -> str:
    """책 코드로부터 제목을 가져옵니다."""
    try:
        with open('backgrounds.json', 'r', encoding='utf-8') as f:
            backgrounds = json.load(f)
        if book_code in backgrounds and 'book' in backgrounds[book_code]:
            return backgrounds[book_code]['book']
    except:
        pass
    
    # 기본 제목
    titles = {
        "SCJ": "심청전",
        "HBJ": "흥부전",
        "BJBJ": "별주부전",
        "OGJJ": "옹고집전",
        "JWCJ": "전우치전",
        "JHHRJ": "장화홍련전",
        "PSJ": "박씨전",
        "DGJ": "두껍전",
        "KWJ": "김원전"
    }
    return titles.get(book_code, "")

def get_characters_for_book(book_code: str, characters_data: dict) -> list:
    """책의 cha1과 cha2 캐릭터 정보를 가져옵니다."""
    if book_code not in ROLE_MAP:
        return []
    
    chars = []
    role_map = ROLE_MAP[book_code]
    
    # cha1과 cha2 가져오기
    for role_key in [role_map["cha1"], role_map["cha2"]]:
        if role_key and book_code in characters_data:
            if role_key in characters_data[book_code]:
                char_data = characters_data[book_code][role_key]
                chars.append({
                    "role_key": role_key,
                    "voice": char_data.get("voice", "alloy"),
                    "speed": 1.0  # 모두 같은 속도로
                })
    
    # 중복 제거 (같은 캐릭터가 cha1과 cha2에 모두 있는 경우, 예: OGJJ, JWCJ)
    seen = set()
    unique_chars = []
    for char in chars:
        key = char["role_key"]
        if key not in seen:
            seen.add(key)
            unique_chars.append(char)
    
    return unique_chars

def generate_tts_for_character(text: str, voice: str, speed: float = 1.0) -> bytes:
    """단일 캐릭터의 TTS를 생성합니다."""
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="wav",
        speed=speed
    )
    return response.read()

def mix_audio_files(audio_files: list, output_path: str):
    """여러 오디오 파일을 믹싱하여 하나로 합칩니다."""
    if not audio_files:
        return False
    
    if len(audio_files) == 1:
        # 파일 하나면 그냥 복사
        subprocess.run(
            ["cp", audio_files[0], output_path],
            check=True
        )
        return True
    
    # ffmpeg로 여러 오디오 믹싱
    # -i 옵션을 여러 번 사용하여 모든 입력 파일 지정
    cmd = ["ffmpeg", "-y"]
    
    # 모든 입력 파일 추가
    for audio_file in audio_files:
        cmd.extend(["-i", audio_file])
    
    # 필터로 믹싱 (모든 채널을 합침)
    filter_complex = "amix=inputs={}:duration=longest".format(len(audio_files))
    cmd.extend(["-filter_complex", filter_complex])
    
    # 출력 설정
    cmd.extend(["-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_path])
    
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        return True
    except Exception as e:
        print(f"⚠️ 오디오 믹싱 실패: {e}")
        return False

def generate_title_chorus(book_code: str, title: str, characters: list):
    """모든 캐릭터가 함께 제목을 외치는 TTS를 생성합니다."""
    output_dir = "title_saying"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{book_code}_title.wav")
    
    # 제목 뒤에 느낌표 추가
    title_with_exclamation = f"{title}!"
    
    print(f"📚 [{book_code}] 제목 생성 중: {title_with_exclamation}")
    print(f"   캐릭터: {', '.join([c['role_key'] for c in characters])} ({len(characters)}명)")
    
    if not characters:
        print(f"⚠️ [{book_code}] 캐릭터가 없습니다.")
        return False
    
    temp_dir = tempfile.gettempdir()
    temp_audio_files = []
    
    try:
        # 각 캐릭터의 TTS 생성
        for i, char in enumerate(characters):
            print(f"   - {char['role_key']} ({char['voice']}) 생성 중...")
            audio_bytes = generate_tts_for_character(
                title_with_exclamation,
                char['voice'],
                char['speed']
            )
            
            # 임시 파일에 저장
            temp_file = os.path.join(temp_dir, f"title_{book_code}_{char['role_key']}_{i}.wav")
            with open(temp_file, "wb") as f:
                f.write(audio_bytes)
            temp_audio_files.append(temp_file)
        
        # 모든 오디오 믹싱
        print(f"   - {len(temp_audio_files)}개 목소리 믹싱 중...")
        if mix_audio_files(temp_audio_files, output_path):
            print(f"✅ [{book_code}] 제목 생성 완료: {output_path}")
            return True
        else:
            print(f"❌ [{book_code}] 오디오 믹싱 실패")
            return False
            
    except Exception as e:
        print(f"❌ [{book_code}] 제목 생성 실패: {e}")
        return False
        
    finally:
        # 임시 파일 정리
        for temp_file in temp_audio_files:
            try:
                os.remove(temp_file)
            except:
                pass

def main():
    """모든 책의 제목을 생성합니다."""
    print("=" * 60)
    print("📚 책 제목 TTS 생성 (모든 캐릭터 합창)")
    print("=" * 60)
    
    # 캐릭터 데이터 로드
    characters_data = load_characters()
    
    success_count = 0
    fail_count = 0
    
    for book_code in ROLE_MAP.keys():
        title = get_book_title(book_code)
        if not title:
            print(f"⚠️ [{book_code}] 제목을 찾을 수 없습니다.")
            fail_count += 1
            continue
        
        characters = get_characters_for_book(book_code, characters_data)
        if not characters:
            print(f"⚠️ [{book_code}] 캐릭터를 찾을 수 없습니다.")
            fail_count += 1
            continue
        
        if generate_title_chorus(book_code, title, characters):
            success_count += 1
        else:
            fail_count += 1
        print()  # 빈 줄
    
    print("=" * 60)
    print(f"✨ 완료: 성공 {success_count}개, 실패 {fail_count}개")
    print("=" * 60)

if __name__ == '__main__':
    main()

