#!/usr/bin/env python3
"""
characters_saying.json의 실제 대사를 심층 분석하여 
characters_tone.json의 말투 정보를 정확하게 업데이트합니다.
"""

import json
import re
from collections import Counter
from typing import Dict, List

def extract_endings_and_expressions(dialogues: List[str]) -> Dict:
    """대사에서 어미와 자주 사용되는 표현을 추출합니다."""
    all_endings = []
    all_expressions = []
    all_words = []
    
    for dialogue in dialogues:
        # 문장을 분리
        sentences = re.split(r'[.!?]\s*', dialogue)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 어미 추출 (문장 끝 부분)
            # ~다, ~요, ~지, ~냐, ~구나, ~니, ~어, ~아, ~오, ~옵니다 등
            ending_match = re.search(r'([다요지냐구나니어아오서소서잉당께지라우옵니다서소서]+)\s*$', sentence)
            if ending_match:
                all_endings.append(ending_match.group(1))
            
            # 자주 사용되는 표현 패턴 추출
            # ~하겠습니다, ~하옵니다, ~하지 마세요, 부디, 틀림없이 등
            patterns = [
                r'~?하겠습니다',
                r'~?하옵니다',
                r'~?하지\s*마세요',
                r'~?하지\s*말아',
                r'부디',
                r'틀림없이',
                r'반드시',
                r'꼭',
                r'제가\s*어떻게든',
                r'제가\s*반드시',
                r'~?하시옵소서',
                r'~?하시지\s*마',
                r'~?하시겠습니까',
                r'~?하느냐',
                r'~?하리라',
                r'~?단다',
                r'~?구나',
                r'~?지라우',
                r'~?당께',
                r'~?요잉',
                r'~?하거라',
                r'~?하라',
                r'이놈',
                r'감히',
                r'죄송',
                r'고맙',
                r'미안',
                r'억울',
                r'슬프',
                r'제발',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                all_expressions.extend(matches)
            
            # 단어 추출
            words = sentence.split()
            all_words.extend(words)
    
    # 빈도수 계산
    ending_counter = Counter(all_endings)
    expression_counter = Counter(all_expressions)
    word_counter = Counter(all_words)
    
    return {
        "top_endings": [item[0] for item in ending_counter.most_common(15)],
        "top_expressions": [item[0] for item in expression_counter.most_common(25)],
        "top_words": [item[0] for item in word_counter.most_common(20)]
    }

def analyze_characteristics(dialogues: List[str]) -> Dict:
    """대사에서 말투 특징을 심층 분석합니다."""
    if not dialogues:
        return {}
    
    full_text = ' '.join(dialogues)
    
    characteristics = {
        "formality_indicators": [],
        "politeness_indicators": [],
        "emotional_keywords": [],
        "dialect_indicators": [],
        "age_indicators": [],
        "sentence_patterns": [],
        "repetition_style": []
    }
    
    # 공손도 분석
    very_formal = ['하옵니다', '하시옵소서', '하시옵니까', '하옵소서', '비나이다']
    formal = ['하겠습니다', '하세요', '하시지', '하시면', '하시는']
    informal = ['한다', '한다냐', '하거라', '하라', '하느냐']
    
    for pattern in very_formal:
        if pattern in full_text:
            characteristics["formality_indicators"].append(f"very_formal: {pattern}")
    
    for pattern in formal:
        if pattern in full_text:
            characteristics["politeness_indicators"].append(f"polite: {pattern}")
    
    for pattern in informal:
        if pattern in full_text:
            characteristics["formality_indicators"].append(f"informal: {pattern}")
    
    # 감정 키워드
    emotion_map = {
        "passionate": ['반드시', '틀림없이', '꼭', '제가 어떻게든', '부디', '비나이다', '하겠습니다', '하리라'],
        "sad": ['슬프', '억울', '미안', '불쌍', '한', '원통', '아이고', '흑흑'],
        "angry": ['이놈', '죽일', '화나', '억울해', '분하고', '감히', '가소롭'],
        "fearful": ['무서', '두려', '걱정', '제발', '부디', '안 돼', '못 가'],
        "determined": ['반드시', '틀림없이', '꼭', '하겠', '하리라', '게 섰거라'],
        "gentle": ['~하지 마', '~하지 말아', '부디', '제발', '~하시지 마'],
        "humble": ['죄송', '부족', '못나', '고맙', '감사'],
        "arrogant": ['감히', '어찌', '가소롭', '미련한', '이놈이'],
        "respectful": ['하옵니다', '하시옵소서', '~님', '~께서']
    }
    
    for emotion, keywords in emotion_map.items():
        count = sum(1 for kw in keywords if kw in full_text)
        if count > 0:
            characteristics["emotional_keywords"].append(f"{emotion}: {count}회")
    
    # 방언 특징
    dialect_map = {
        "rural": ['~지라우', '~당께', '~요잉', '~지라', '~하시지라', '아이코', '아이고', '아따'],
        "old_fashioned": ['~하느냐', '~하리라', '~단다', '~구나', '~하옵니다', '~하시옵소서'],
        "modern": ['~거든', '~잖아', '~지 뭐', '~하지', '~하는 거야']
    }
    
    for dialect, patterns in dialect_map.items():
        count = sum(1 for p in patterns if p in full_text)
        if count > 0:
            characteristics["dialect_indicators"].append(f"{dialect}: {count}회")
    
    # 문장 패턴
    if '?' in full_text:
        characteristics["sentence_patterns"].append("질문 많이 사용")
    if '!' in full_text:
        characteristics["sentence_patterns"].append("감탄문 사용")
    if '...' in full_text or '……' in full_text:
        characteristics["sentence_patterns"].append("말줄임표 사용 (감정 표현)")
    
    # 반복 스타일
    if re.search(r'(.)\1{2,}', full_text):
        characteristics["repetition_style"].append("단어 반복 (강조)")
    
    # 평균 문장 길이
    sentences = re.split(r'[.!?]\s*', full_text)
    avg_length = sum(len(s.split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()]) if sentences else 0
    characteristics["sentence_patterns"].append(f"평균 문장 길이: {avg_length:.1f}단어")
    
    return characteristics

def update_tone_from_sayings():
    """characters_saying.json을 읽어서 characters_tone.json을 업데이트합니다."""
    
    # 파일 읽기
    with open('characters_saying.json', 'r', encoding='utf-8') as f:
        sayings_data = json.load(f)
    
    with open('characters_tone.json', 'r', encoding='utf-8') as f:
        tone_data = json.load(f)
    
    # 각 캐릭터별로 분석 및 업데이트
    for book_code, characters in sayings_data.items():
        if book_code not in tone_data:
            continue
        
        for char_key, char_data in characters.items():
            if char_key not in tone_data[book_code]:
                continue
            
            dialogues = char_data.get('dialogues', [])
            if not dialogues:
                continue
            
            print(f"\n📝 분석 중: {book_code} - {char_key} ({len(dialogues)}개 대사)")
            
            # 말투 패턴 분석
            patterns = extract_endings_and_expressions(dialogues)
            characteristics = analyze_characteristics(dialogues)
            
            # 기존 데이터 업데이트
            char_tone = tone_data[book_code][char_key]
            
            # speech_patterns 업데이트
            if 'speech_patterns' not in char_tone:
                char_tone['speech_patterns'] = {}
            
            # 기존 frequent_expressions와 병합
            existing_expressions = char_tone.get('speech_patterns', {}).get('frequent_expressions', [])
            new_expressions = patterns.get('top_expressions', [])
            
            # 중복 제거하고 병합 (실제 대사에서 추출한 것을 우선)
            merged_expressions = []
            seen = set()
            
            # 새로 추출한 표현을 먼저 추가
            for expr in new_expressions:
                if expr not in seen:
                    merged_expressions.append(expr)
                    seen.add(expr)
            
            # 기존 표현 중 아직 추가되지 않은 것만 추가
            for expr in existing_expressions:
                if expr not in seen:
                    merged_expressions.append(expr)
                    seen.add(expr)
            
            char_tone['speech_patterns']['frequent_expressions'] = merged_expressions[:30]
            
            # 추가 정보 저장
            char_tone['speech_patterns']['endings_from_dialogues'] = patterns.get('top_endings', [])[:15]
            char_tone['speech_patterns']['common_words'] = patterns.get('top_words', [])[:15]
            
            # 분석 결과를 새로운 필드로 추가
            char_tone['analysis_from_dialogues'] = {
                "formality_indicators": characteristics.get('formality_indicators', [])[:5],
                "politeness_indicators": characteristics.get('politeness_indicators', [])[:5],
                "emotional_keywords": characteristics.get('emotional_keywords', [])[:8],
                "dialect_indicators": characteristics.get('dialect_indicators', []),
                "sentence_patterns": characteristics.get('sentence_patterns', []),
                "repetition_style": characteristics.get('repetition_style', [])
            }
            
            print(f"✅ 업데이트 완료: {char_key}")
            print(f"   - 자주 사용 표현: {len(merged_expressions)}개")
            print(f"   - 어미 패턴: {len(patterns.get('top_endings', []))}개")
            print(f"   - 감정 키워드: {len(characteristics.get('emotional_keywords', []))}개")
            if characteristics.get('dialect_indicators'):
                print(f"   - 방언 특징: {', '.join([d.split(':')[0] for d in characteristics['dialect_indicators']])}")
    
    # 업데이트된 데이터 저장
    with open('characters_tone.json', 'w', encoding='utf-8') as f:
        json.dump(tone_data, f, ensure_ascii=False, indent=2)
    
    print("\n✨ characters_tone.json 업데이트 완료!")

if __name__ == '__main__':
    update_tone_from_sayings()

