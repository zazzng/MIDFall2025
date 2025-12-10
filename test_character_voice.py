#!/usr/bin/env python3
"""
캐릭터 목소리 테스트 스크립트
각 캐릭터가 "안녕하세요"라고 말하는 것을 들어볼 수 있습니다.
"""

import os
import json
import subprocess
import tempfile
import re
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

# tts.py에서 공통 함수 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts import apply_audio_effects

# 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = "gpt-4o-mini-tts"
ALLOWED_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
}

def load_characters():
    """characters_tone.json 로드"""
    with open("characters_tone.json", "r", encoding="utf-8") as f:
        return json.load(f)

def list_all_characters(characters):
    """모든 캐릭터 목록 출력"""
    print("\n📋 사용 가능한 캐릭터 목록:\n")
    for book_code, roles in characters.items():
        print(f"  [{book_code}]")
        for role_key, char_data in roles.items():
            gender = char_data.get("gender", "unknown")
            age = char_data.get("age", "?")
            voice = char_data.get("voice", "alloy")
            print(f"    - {role_key} ({gender}, {age}세, voice: {voice})")
    print()

def find_character(characters, name):
    """캐릭터 이름으로 찾기 (book_code-role_key 형식 또는 role_key만)"""
    name = name.strip().lower()
    
    # book_code-role_key 형식인 경우
    if "-" in name:
        parts = name.split("-", 1)
        book_code = parts[0].upper()
        role_key = parts[1]
        if book_code in characters and role_key in characters[book_code]:
            return book_code, role_key, characters[book_code][role_key]
    
    # role_key만 입력한 경우 - 모든 책에서 검색
    for book_code, roles in characters.items():
        if name in roles:
            return book_code, name, roles[name]
    
    return None, None, None

def generate_test_tts(character, book_code, role_key, text="안녕하세요"):
    """캐릭터의 목소리로 TTS 생성 및 재생"""
    voice = character.get("voice", "alloy")
    speed = character.get("speed", 1.0)
    
    if voice not in ALLOWED_VOICES:
        voice = "alloy"
    
    print(f"🎤 [{book_code}-{role_key}] '{text}' 생성 중... (voice: {voice}, speed: {speed})")
    
    # TTS 생성
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=voice,
        input=text,
        response_format="wav",
        speed=speed
    )
    
    audio_bytes = response.read()
    
    # 임시 파일에 원본 오디오 저장
    temp_dir = tempfile.gettempdir()
    temp_input = os.path.join(temp_dir, f"test_voice_{os.getpid()}.wav")
    with open(temp_input, "wb") as f:
        f.write(audio_bytes)
    
    # 특수 캐릭터 오디오 효과 적용 (tts.py의 공통 함수 사용)
    output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
    
    # 캐릭터 정보 딕셔너리 생성 (apply_audio_effects 함수에 필요한 형식)
    char_dict = {
        "book_code": book_code,
        "role_key": role_key
    }
    
    # 공통 함수로 오디오 효과 적용
    try:
        apply_audio_effects(char_dict, temp_input, output_path)
        
        # 파일이 완전히 생성될 때까지 대기 (최대 3초)
        max_wait = 3.0
        wait_interval = 0.1
        waited = 0.0
        while waited < max_wait:
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                # 파일이 안정화될 때까지 조금 더 대기
                time.sleep(0.1)
                break
            time.sleep(wait_interval)
            waited += wait_interval
        
        # 파일이 제대로 생성되었는지 확인
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print(f"⚠️  오디오 효과 적용 실패, 원본 파일 사용")
            output_path = temp_input
    except Exception as e:
        print(f"⚠️  오디오 효과 적용 중 오류 발생: {e}")
        print(f"    원본 파일로 재생합니다.")
        output_path = temp_input
    
    # 오디오 재생
    print(f"🔊 재생 중...")
    try:
        # 파일이 존재하고 크기가 0이 아닌지 확인
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            subprocess.run(["afplay", output_path], check=True, timeout=30)
        else:
            print(f"❌ 오디오 파일이 생성되지 않았습니다.")
            return
    except subprocess.CalledProcessError as e:
        print(f"❌ 오디오 재생 실패: {e}")
        return
    except Exception as e:
        print(f"❌ 오디오 재생 중 오류 발생: {e}")
        return
    
    # 임시 파일 삭제
    try:
        if output_path != temp_input and os.path.exists(output_path):
            os.remove(output_path)
        if os.path.exists(temp_input):
            os.remove(temp_input)
    except:
        pass
    
    print(f"✅ 완료!\n")

def main():
    """메인 함수"""
    characters = load_characters()
    
    print("=" * 60)
    print("🎭 캐릭터 목소리 테스트")
    print("=" * 60)
    print("\n각 캐릭터가 '안녕하세요'라고 말하는 것을 들어볼 수 있습니다.")
    print("캐릭터 이름을 입력하세요 (예: simcheong, SCJ-simcheong)")
    print("'list'를 입력하면 전체 목록을 볼 수 있습니다.")
    print("'quit' 또는 'q'를 입력하면 종료합니다.\n")
    
    while True:
        try:
            user_input = input("캐릭터 이름 입력 (또는 'list'/'quit'): ")
            # 모든 제어 문자 제거 (캐리지 리턴, 줄바꿈 등)
            user_input = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', user_input)  # 모든 제어 문자 제거
            user_input = user_input.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        
        if not user_input:
            continue
        
        user_input_lower = user_input.lower().strip()
        
        # 디버깅: 입력값 확인
        if user_input != user_input_lower:
            print(f"[DEBUG] 입력값: {repr(user_input)}")
        
        if user_input_lower in ['quit', 'q', 'exit']:
            print("👋 종료합니다.")
            break
        
        if user_input_lower == 'list':
            list_all_characters(characters)
            continue
        
        # 캐릭터 찾기
        book_code, role_key, char_data = find_character(characters, user_input)
        
        if char_data is None:
            print(f"❌ '{user_input}' 캐릭터를 찾을 수 없습니다.")
            print("   'list'를 입력하여 전체 목록을 확인하세요.\n")
            continue
        
        # 사용자 정의 텍스트 입력 받기 (선택사항)
        try:
            custom_text = input(f"말할 내용 (기본값: '안녕하세요', Enter로 기본값 사용): ")
            custom_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', custom_text)  # 모든 제어 문자 제거
            custom_text = custom_text.strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 종료합니다.")
            break
        if not custom_text:
            custom_text = "안녕하세요"
        
        # TTS 생성 및 재생
        try:
            generate_test_tts(char_data, book_code, role_key, custom_text)
        except Exception as e:
            print(f"❌ 오류 발생: {e}\n")

if __name__ == "__main__":
    main()

