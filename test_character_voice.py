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
from openai import OpenAI
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TTS_MODEL = "gpt-4o-mini-tts"
ALLOWED_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"
}

def load_characters():
    """characters.json 로드"""
    with open("characters.json", "r", encoding="utf-8") as f:
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
    
    # 특수 캐릭터 오디오 효과 적용 (tts.py와 동일한 로직)
    output_path = temp_input
    
    if (book_code == "JHHRJ" and role_key == "ghost") or (book_code == "KWJ" and role_key == "monster"):
        # reverb 효과 적용 (aecho 필터 사용)
        # ghost의 경우: 더 서글프고 울먹거리는 효과를 위해 tremolo와 pitch 조정도 추가
        if book_code == "JHHRJ" and role_key == "ghost":
            # ghost: 구슬프고 우울하고 한이 서린 처녀귀신 목소리
            # 효과: 깊은 reverb + 강한 tremolo (울먹거림) + 낮은 pitch (어둡고 우울) + 느린 속도 + 고주파 필터링 + delay + equalizer
            output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
            # 오디오 필터 체인을 하나의 문자열로 합침
            audio_filter = (
                "asetrate=44100*0.92,aresample=44100,"  # pitch 낮춤 (더 어둡고 우울하게)
                "atempo=0.95,"  # 속도 느리게 (더 구슬프게)
                "lowpass=f=3000,"  # 고주파 필터링 (더 어둡고 깊게)
                "aecho=1.0:0.9:120:0.5,"  # 깊고 긴 reverb (더 공허하고 처절하게)
                "adelay=50|50,"  # 약간의 delay (에코 효과)
                "tremolo=f=3.0:d=0.4,"  # 매우 강한 tremolo (더욱 울먹거리게)
                "equalizer=f=200:width_type=h:width=300:g=2,"  # 저주파 강조 (더 깊고 묵직하게)
                "equalizer=f=5000:width_type=h:width=2000:g=-3"  # 고주파 억제 (더 어둡게)
            )
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", audio_filter,
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        else:
            # monster: reverb만 적용
            output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "aecho=0.8:0.88:60:0.4",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
        try:
            os.remove(temp_input)
        except:
            pass
    elif book_code == "SCJ" and role_key == "simcheong":
        # 심청: 어리고 명랑하고 결연에 가득 찬 목소리
        # 효과: 높은 pitch (어리고 밝게) + 빠른 속도 (명랑함) + 고주파 강조 (맑고 밝게) + vibrato (생동감) + 저주파 억제 (가볍고 밝게)
        output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
        # 오디오 필터 체인을 하나의 문자열로 합침
        audio_filter = (
            "asetrate=44100*1.12,aresample=44100,"  # pitch 올림 (더 어리고 밝게)
            "atempo=1.08,"  # 속도 빠르게 (명랑하고 활기차게)
            "equalizer=f=3000:width_type=h:width=2000:g=3,"  # 고주파 강조 (맑고 밝게)
            "equalizer=f=5000:width_type=h:width=1500:g=2,"  # 더 높은 고주파 강조 (명랑함)
            "equalizer=f=200:width_type=h:width=300:g=-2,"  # 저주파 억제 (가볍고 밝게)
            "vibrato=f=5.5:d=0.15,"  # 약간의 vibrato (생동감과 결연함)
            "highpass=f=100"  # 매우 낮은 주파수 제거 (더 맑게)
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp_input,
             "-af", audio_filter,
             output_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        try:
            os.remove(temp_input)
        except:
            pass
    elif book_code == "DGJ":
        if role_key == "fox":
            # 여우: 교활하고 가는 목소리, 간신배 느낌
            # pitch를 약간 올려서 더 가늘게, tremolo를 약간 추가해서 교활한 느낌
            output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "asetrate=44100*1.15,aresample=44100,tremolo=f=3.0:d=0.2",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            try:
                os.remove(temp_input)
            except:
                pass
        elif role_key == "toad":
            # 두꺼비: 현명하고 총명하고 뭉툭하고 묵직한 목소리
            # pitch를 약간 낮춰서 더 묵직하게, bass boost로 더 깊고 뭉툭한 느낌
            output_path = os.path.join(temp_dir, f"test_voice_processed_{os.getpid()}.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_input,
                 "-af", "asetrate=44100*0.9,aresample=44100,equalizer=f=100:width_type=h:width=200:g=3",
                 output_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            try:
                os.remove(temp_input)
            except:
                pass
    
    # 오디오 재생
    print(f"🔊 재생 중...")
    subprocess.run(["afplay", output_path])
    
    # 임시 파일 삭제
    try:
        os.remove(output_path)
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

