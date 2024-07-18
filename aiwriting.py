import jpype
import os
import streamlit as st
from PyKakao import KoGPT
from konlpy.tag import Okt
import pandas as pd

# JVM 초기화 함수
def initialize_jvm():
    konlpy_java_path = os.path.join(os.path.dirname(__file__), "C:\\Python\\lib\\site-packages\\konlpy\\java\\*")
    if not jpype.isJVMStarted():
        jpype.startJVM(jpype.getDefaultJVMPath(), f'-Djava.class.path={konlpy_java_path}', '-ea')

# JVM 초기화
initialize_jvm()

# KoGPT API 키 설정
api = KoGPT(service_key="4fc12938380daae98223c2e26db8e3cd")
okt = Okt()

# 금칙어 데이터 로드
@st.cache_data
def load_badwords():
    df = pd.read_csv("static/badword.csv", encoding='euc-kr')
    return df['word'].tolist()

badwords = load_badwords()

# 금칙어 체크 함수
def check_badwords(text):
    words = text.split()
    badwords_found = [word for word in words if word in badwords]
    return badwords_found

# 문장 생성 함수
def generate_text(prompt, max_tokens=64):
    result = api.generate(prompt, max_tokens, temperature=0.7, top_p=0.8)
    return result['generations'][0]['text']

# Streamlit UI 설정
st.title("인공지능 작문 생성기")
st.write("이야기를 이어가고 싶은 문장을 입력하세요.")

user_input = st.text_area("입력 문장", height=300)
if st.button("생성 시작"):
    if user_input:
        with st.spinner("문장을 생성 중입니다..."):
            prompt = user_input[-350:]
            rest_text = user_input[:-350] if len(user_input) > 350 else ''

            generated_text = generate_text(prompt, 32)
            complete_text = rest_text + prompt + generated_text

            badwords_found = check_badwords(complete_text)

            if badwords_found:
                st.error(f"부적절한 단어가 포함되어 있습니다: {', '.join(badwords_found)}")
            else:
                st.success("문장 생성 완료!")
                st.text_area("생성된 문장", complete_text, height=300)

                nouns = okt.nouns(complete_text)
                st.write("추출된 명사:", ', '.join(nouns))
    else:
        st.warning("문장을 입력하세요.")
