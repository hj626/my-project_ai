import pandas as pd
import os

folder_path = r'C:\Users\human-24\Downloads\lung'
files_to_convert = ['Fusion_Lung', 'INDEL_Lung', 'SNV_Lung']

for file_name in files_to_convert:
    # 파일 확장자 .txt를 추가함
    txt_file = os.path.join(folder_path, file_name + '.txt')
    csv_file = os.path.join(folder_path, file_name + '.csv')

    try:
        # 탭 구분자 파일 읽기
        df = pd.read_csv(txt_file, sep='\t', encoding='utf-8')
        
        num_columns = df.shape[1]
        print(f"'{file_name}' 파일의 컬럼 개수: {num_columns}개")

        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"변환 완료: {csv_file}")
    except FileNotFoundError:
        print(f"오류: '{txt_file}' 파일을 찾을 수 없습니다. (확장자 .txt가 붙어있는지 확인하세요)")
    except Exception as e:
        print(f"변환 중 오류 발생: {e}")