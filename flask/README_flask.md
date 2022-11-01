7팀 기업과제3 - KLUE/STS

REST.API 구현 파일입니다.

사용된 라이브러리
pandas==1.1.5
torch==1.9.0
transformers==4.15.0
Flask==2.0.1

사용 순서
1. local에서 conda 환경설정을 합니다. (conda create -n '환경이름' python=3.6)
2. conda를 실행합니다.
3. pip install -r requirements.txt를 실행하여 필요한 라이브러리를 설치합니다.
4. __init__.py에서 model = MODEL('가중치 저장위치', '사용할 tokenizer')를 알맞게 설정해줍니다.
5. __init__.py를 실행하여 로컬 127.0.0.1:5000 에서 실행되는지 확인합니다.
6. 실행창에서 빈칸에 문장을 입력해줍니다.
7. 검색하기 버튼을 클릭하면 다음 페이지로 넘어가며 결과값이 제대로 출력되는지 확인합니다.