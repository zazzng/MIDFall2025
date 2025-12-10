#!/usr/bin/env python3
"""
characters_saying.json의 실제 대사를 분석하여 
characters_tone.json의 말투 정보를 업데이트합니다.
"""

import json
import re
from collections import Counter
from typing import Dict, List, Set

def analyze_speech_patterns(dialogues: List[str]) -> Dict:
    """대사 목록을 분석하여 말투 패턴을 추출합니다."""
    if not dialogues:
        return {}
    
    # 어미 패턴 추출
    endings = []
    frequent_expressions = []
    sentence_starters = []
    
    for dialogue in dialogues:
        # 문장 끝 어미 추출
        # ~다, ~요, ~지, ~냐, ~구나, ~니, ~어, ~아 등
        endings_match = re.findall(r'[~]?([다요지냐구나니어아오옵니다서소서잉당께지라우]+\s*[.!?]?)', dialogue)
        if endings_match:
            endings.extend(endings_match)
        
        # 자주 사용되는 표현 추출
        # ~하겠습니다, ~하옵니다, ~하지 마세요 등
        expressions = re.findall(r'[~]?([하겠습니하옵니하지마시부디틀림없반드시꼭제가]+\s*[다요지냐구나니어아오서소서잉당께지라우]*)', dialogue)
        frequent_expressions.extend(expressions)
        
        # 문장 시작 패턴
        if dialogue.strip():
            first_words = dialogue.strip().split()[:3]
            if first_words:
                sentence_starters.append(' '.join(first_words))
    
    # 빈도수 계산
    ending_counter = Counter(endings)
    expression_counter = Counter(frequent_expressions)
    starter_counter = Counter(sentence_starters)
    
    # 상위 20개 추출
    top_endings = [item[0] for item in ending_counter.most_common(20)]
    top_expressions = [item[0] for item in expression_counter.most_common(20)]
    top_starters = [item[0] for item in starter_counter.most_common(10)]
    
    return {
        "endings": top_endings,
        "frequent_expressions": top_expressions,
        "sentence_starters": top_starters
    }

def extract_tone_characteristics(dialogues: List[str]) -> Dict:
    """대사에서 말투 특징을 추출합니다."""
    if not dialogues:
        return {}
    
    characteristics = {
        "formality_level": "unknown",  # formal, informal, mixed
        "politeness_level": "unknown",  # very_polite, polite, casual, rude
        "emotional_tone": [],  # passionate, calm, sad, angry, etc.
        "dialect_features": [],  # regional dialect patterns
        "age_indicators": [],  # old-fashioned, modern, etc.
        "sentence_length": "unknown",  # short, medium, long
        "repetition_patterns": []
    }
    
    # 공손어미 분석
    formal_endings = ['하옵니다', '하시옵소서', '하시옵니까', '하옵소서']
    informal_endings = ['한다', '한다냐', '하거라', '하라']
    polite_endings = ['하겠습니다', '하세요', '하시지', '하시면']
    
    formal_count = sum(1 for d in dialogues if any(e in d for e in formal_endings))
    informal_count = sum(1 for d in dialogues if any(e in d for e in informal_endings))
    polite_count = sum(1 for d in dialogues if any(e in d for e in polite_endings))
    
    if formal_count > len(dialogues) * 0.5:
        characteristics["formality_level"] = "formal"
        characteristics["politeness_level"] = "very_polite"
    elif polite_count > len(dialogues) * 0.5:
        characteristics["formality_level"] = "polite"
        characteristics["politeness_level"] = "polite"
    elif informal_count > len(dialogues) * 0.5:
        characteristics["formality_level"] = "informal"
        characteristics["politeness_level"] = "casual"
    else:
        characteristics["formality_level"] = "mixed"
        characteristics["politeness_level"] = "mixed"
    
    # 감정 톤 분석
    emotional_keywords = {
        "passionate": ["반드시", "틀림없이", "꼭", "제가 어떻게든", "부디", "비나이다"],
        "sad": ["슬프", "억울", "미안", "불쌍", "한", "원통"],
        "angry": ["이놈", "죽일", "화나", "억울해", "분하고"],
        "fearful": ["무서", "두려", "걱정", "제발", "부디"],
        "determined": ["반드시", "틀림없이", "꼭", "하겠", "하리라"],
        "gentle": ["~하지 마", "~하지 말아", "부디", "제발"],
        "arrogant": ["감히", "어찌", "가소롭", "미련한"],
        "humble": ["죄송", "부족", "못나", "고맙"]
    }
    
    for emotion, keywords in emotional_keywords.items():
        count = sum(1 for d in dialogues if any(k in d for k in keywords))
        if count > len(dialogues) * 0.2:
            characteristics["emotional_tone"].append(emotion)
    
    # 방언 특징 분석
    dialect_patterns = {
        "rural": ["~지라우", "~당께", "~요잉", "~지라", "~하시지라"],
        "old_fashioned": ["~하느냐", "~하리라", "~단다", "~구나", "~하옵니다"],
        "modern": ["~거든", "~잖아", "~지 뭐", "~하지"]
    }
    
    for dialect, patterns in dialect_patterns.items():
        count = sum(1 for d in dialogues if any(p in d for p in patterns))
        if count > len(dialogues) * 0.2:
            characteristics["dialect_features"].append(dialect)
    
    # 문장 길이 분석
    avg_length = sum(len(d.split()) for d in dialogues) / len(dialogues) if dialogues else 0
    if avg_length < 5:
        characteristics["sentence_length"] = "short"
    elif avg_length < 10:
        characteristics["sentence_length"] = "medium"
    else:
        characteristics["sentence_length"] = "long"
    
    # 반복 패턴 분석
    for dialogue in dialogues:
        # 반복되는 어미나 표현 찾기
        repeated = re.findall(r'(.{2,5})\1+', dialogue)
        if repeated:
            characteristics["repetition_patterns"].extend(repeated)
    
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
            patterns = analyze_speech_patterns(dialogues)
            characteristics = extract_tone_characteristics(dialogues)
            
            # 기존 데이터 업데이트
            char_tone = tone_data[book_code][char_key]
            
            # frequent_expressions 업데이트 (실제 대사에서 추출한 것과 기존 것 병합)
            existing_expressions = char_tone.get('speech_patterns', {}).get('frequent_expressions', [])
            new_expressions = patterns.get('frequent_expressions', [])
            
            # 중복 제거하고 병합
            merged_expressions = list(dict.fromkeys(existing_expressions + new_expressions))[:30]
            
            # speech_patterns 업데이트
            if 'speech_patterns' not in char_tone:
                char_tone['speech_patterns'] = {}
            
            char_tone['speech_patterns']['frequent_expressions'] = merged_expressions
            char_tone['speech_patterns']['endings'] = patterns.get('endings', [])[:20]
            char_tone['speech_patterns']['sentence_starters'] = patterns.get('sentence_starters', [])[:10]
            
            # tone_characteristics에 분석 결과 추가
            if 'tone_characteristics' not in char_tone:
                char_tone['tone_characteristics'] = ""
            
            # 기존 tone_characteristics에 분석된 특징 추가
            existing_tone = char_tone['tone_characteristics']
            
            # 분석 결과를 텍스트로 추가
            analysis_text = f"\n\n[실제 대사 분석 결과]\n"
            analysis_text += f"- 공손도: {characteristics['politeness_level']}\n"
            analysis_text += f"- 격식: {characteristics['formality_level']}\n"
            analysis_text += f"- 감정 톤: {', '.join(characteristics['emotional_tone']) if characteristics['emotional_tone'] else '없음'}\n"
            analysis_text += f"- 방언 특징: {', '.join(characteristics['dialect_features']) if characteristics['dialect_features'] else '없음'}\n"
            analysis_text += f"- 문장 길이: {characteristics['sentence_length']}\n"
            
            if characteristics['repetition_patterns']:
                analysis_text += f"- 반복 패턴: {', '.join(set(characteristics['repetition_patterns'][:5]))}\n"
            
            char_tone['tone_characteristics'] = existing_tone + analysis_text
            
            print(f"✅ 업데이트 완료: {char_key}")
            print(f"   - 자주 사용 표현: {len(merged_expressions)}개")
            print(f"   - 어미 패턴: {len(patterns.get('endings', []))}개")
            print(f"   - 감정 톤: {', '.join(characteristics['emotional_tone'][:3]) if characteristics['emotional_tone'] else '없음'}")
    
    # 업데이트된 데이터 저장
    with open('characters_tone.json', 'w', encoding='utf-8') as f:
        json.dump(tone_data, f, ensure_ascii=False, indent=2)
    
    print("\n✨ characters_tone.json 업데이트 완료!")

if __name__ == '__main__':
    update_tone_from_sayings()

