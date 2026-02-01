import pymysql
from pymysql import Error
import os
from dotenv import load_dotenv

# .env 파일에 정의된 환경 변수를 로드
load_dotenv()

def __init__(self):
        self.connection = None
        try:
            self.connection = pymysql.connect(
                host='mariadb',
                port=3306,
                
                database='test',  # test 데이터베이스 사용
                user='root',
                password='123456',  # mariadb 설치 당시의 패스워드, 실제 환경에서는 보안을 위해 환경변수 등을 사용
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor   # 쿼리 결과를 딕셔너리로 변환
            )
            print("MariaDB에 성공적으로 연결되었습니다.")
        except Error as e:
            print(f"MariaDB 연결 중 오류 발생: {e}")

# def get_connection():
    
#     conn = pymysql.connect(
#         host=os.getenv("DB_HOST"),        # DB 주소
#         user=os.getenv("DB_USER"),        # DB 사용자명
#         password=os.getenv("DB_PASSWORD"),# DB 비밀번호 (노출 X)
#         database=os.getenv("DB_NAME"),    # DB 이름
#         cursorclass=pymysql.cursors.DictCursor
#     )
    
#     print(" 데이터베이스 연결 성공!")
#     return conn
