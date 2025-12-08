#!/usr/bin/env python3
"""
각 책의 character1과 character2가 함께 제목을 말하는 .wav 파일을 생성합니다.
"""
import os
import json
import subprocess
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = "gpt-4o-mini-tts"

ALLOWED_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
}

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

CHARACTERS = load_json("characters.json")
BACKGROUNDS = load_json("backgrounds.json")

ROLE_MAP = {
    "SCJ": {"cha1": "simcheong",    "cha2": "simbongsa"},
    "HBJ": {"cha1": "heungbu",      "cha2": "nolbu"},
    "BJBJ": {"cha1": "turtle",      "cha2": "rabbit"},
    "OGJJ": {"cha1": "onggojip",    "cha2": "onggojip"},
    "JWCJ": {"cha1": "jeonwoochi",  "cha2": "jeonwoochi"},
    "JHHRJ": {"cha1": "sister_older",    "cha2": "sister_younger"},
    "PSJ": {"cha1": "ugly",         "cha2": "pretty"},
    "DGJ": {"cha1": "toad",         "cha2": "fox"},
    "KWJ": {"cha1": "kimwon",       "cha2": "monster"}
}

def build_character(book_code: str, role_key: str) -> dict:
    data = CHARACTERS[book_code][role_key]
    
    gender = data["gender"]
    age = data["age"]
    base_desc = data["base_personality"]
    raw_voice = data.get("voice", "alloy")
    voice = raw_voice if raw_voice in ALLOWED_VOICES else "alloy"
    
    personality = f"{base_desc}"
    
    return {
        "book_code": book_code,
        "role_key": role_key,
        "gender": gender,
        "age": age,
        "voice": voice,
        "personality": personality,
    }

def generate_tts(character: dict, text: str, output_path: str, speed: float = None):
    """TTS 생성"""
    voice = character.get("voice", "alloy")
    if speed is None:
        speed = CHARACTERS.get(character['book_code'], {}).get(character['role_key'], {}).get('speed', 1.0)
    
    print(f"🎤 [{character['book_code']}-{character['role_key']}] Generating: {text} (speed: {speed})")
    
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        speed=speed
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)

def mix_audio_files(files: list, output: str):
    """ffmpeg를 사용하여 여러 오디오 파일을 믹싱"""
    try:
        inputs = []
        filter_inputs = []
        for i, file in enumerate(files):
            inputs.extend(["-i", file])
            filter_inputs.append(f"[{i}:a]")
        
        filter_complex = f"{''.join(filter_inputs)}amix=inputs={len(files)}:duration=longest:dropout_transition=0"
        
        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-ac", "2",
            output
        ]
        
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✅ Mixed {len(files)} files: {output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error mixing audio: {e}")
        raise

def main():
    # title_saying 폴더 생성
    os.makedirs("title_saying", exist_ok=True)
    
    # 임시 파일 저장 폴더
    os.makedirs("title_saying/temp", exist_ok=True)
    
    for book_code, bg_info in BACKGROUNDS.items():
        book_title = bg_info.get("book", "")
        if not book_title:
            print(f"⚠️ {book_code}: 책 제목이 없습니다.")
            continue
        
        if book_code not in ROLE_MAP:
            print(f"⚠️ {book_code}: ROLE_MAP에 없습니다.")
            continue
        
        # 애니메이션 스타일 제목 (느낌표 추가)
        # 장화홍련전과 김원전은 느낌표를 더 강조
        if book_code == "JHHRJ" or book_code == "KWJ":
            title_text = f"{book_title}!!"
        else:
            title_text = f"{book_title}!"
        
        print(f"\n📚 {book_code}: {title_text}")
        
        # 특수 케이스 처리
        if book_code == "JHHRJ":
            # 장화홍련전: 2명 (sister_older, sister_younger)
            cha1_role = ROLE_MAP[book_code]["cha1"]
            cha2_role = ROLE_MAP[book_code]["cha2"]
            
            cha1 = build_character(book_code, cha1_role)
            cha2 = build_character(book_code, cha2_role)
            
            # 속도 평균 계산
            speed1 = CHARACTERS[book_code][cha1_role].get('speed', 1.0)
            speed2 = CHARACTERS[book_code][cha2_role].get('speed', 1.0)
            avg_speed = (speed1 + speed2) / 2.0
            
            print(f"   cha1: {cha1_role}, cha2: {cha2_role}")
            print(f"   평균 속도: {avg_speed:.2f}")
            
            temp_cha1 = f"title_saying/temp/{book_code}_cha1_title.wav"
            temp_cha2 = f"title_saying/temp/{book_code}_cha2_title.wav"
            
            generate_tts(cha1, title_text, temp_cha1, speed=avg_speed)
            generate_tts(cha2, title_text, temp_cha2, speed=avg_speed)
            
            final_output = f"title_saying/{book_code}_title.wav"
            mix_audio_files([temp_cha1, temp_cha2], final_output)
            
            os.remove(temp_cha1)
            os.remove(temp_cha2)
            
        elif book_code == "JWCJ":
            # 전우치전: 2명이서 말함 (cha1과 cha2, 둘 다 jeonwoochi)
            cha1_role = ROLE_MAP[book_code]["cha1"]
            cha2_role = ROLE_MAP[book_code]["cha2"]
            
            cha1 = build_character(book_code, cha1_role)
            cha2 = build_character(book_code, cha2_role)
            
            # 속도 평균 계산
            speed1 = CHARACTERS[book_code][cha1_role].get('speed', 1.0)
            speed2 = CHARACTERS[book_code][cha2_role].get('speed', 1.0)
            avg_speed = (speed1 + speed2) / 2.0
            
            print(f"   cha1: {cha1_role}, cha2: {cha2_role}")
            print(f"   평균 속도: {avg_speed:.2f}")
            
            temp_cha1 = f"title_saying/temp/{book_code}_cha1_title.wav"
            temp_cha2 = f"title_saying/temp/{book_code}_cha2_title.wav"
            
            generate_tts(cha1, title_text, temp_cha1, speed=avg_speed)
            generate_tts(cha2, title_text, temp_cha2, speed=avg_speed)
            
            final_output = f"title_saying/{book_code}_title.wav"
            mix_audio_files([temp_cha1, temp_cha2], final_output)
            
            os.remove(temp_cha1)
            os.remove(temp_cha2)
            
        else:
            # 일반 케이스: cha1과 cha2
            cha1_role = ROLE_MAP[book_code]["cha1"]
            cha2_role = ROLE_MAP[book_code]["cha2"]
            
            cha1 = build_character(book_code, cha1_role)
            cha2 = build_character(book_code, cha2_role)
            
            # 속도 평균 계산
            speed1 = CHARACTERS[book_code][cha1_role].get('speed', 1.0)
            speed2 = CHARACTERS[book_code][cha2_role].get('speed', 1.0)
            avg_speed = (speed1 + speed2) / 2.0
            
            # 옹고집전은 느리게 조정
            if book_code == "OGJJ":
                avg_speed = 0.8
            
            print(f"   cha1: {cha1_role}, cha2: {cha2_role}")
            print(f"   평균 속도: {avg_speed:.2f}")
            
            temp_cha1 = f"title_saying/temp/{book_code}_cha1_title.wav"
            temp_cha2 = f"title_saying/temp/{book_code}_cha2_title.wav"
            
            generate_tts(cha1, title_text, temp_cha1, speed=avg_speed)
            generate_tts(cha2, title_text, temp_cha2, speed=avg_speed)
            
            final_output = f"title_saying/{book_code}_title.wav"
            mix_audio_files([temp_cha1, temp_cha2], final_output)
            
            os.remove(temp_cha1)
            os.remove(temp_cha2)
    
    # 임시 폴더 삭제
    os.rmdir("title_saying/temp")
    print("\n✅ 모든 제목 말하기 파일 생성 완료!")

if __name__ == "__main__":
    main()

