import pandas as pd
import os

# 파일 경로 설정
folder_path = r'C:\Users\human-24\Downloads\lung'
files_to_merge = ['Fusion_Lung.csv', 'INDEL_Lung.csv', 'SNV_Lung.csv']
output_parquet = os.path.join(folder_path, 'Combined_Lung_Data.parquet')

all_df = []

for file_name in files_to_merge:
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # 각 CSV 파일을 읽어 리스트에 저장
        df = pd.read_csv(file_path, encoding='utf-8')
        
        # 어떤 종류의 데이터인지 구분하기 위한 'data_type' 컬럼 추가 (필요 시)
        df['source_file'] = file_name.split('.')[0]
        
        all_df.append(df)
        print(f"'{file_name}' 데이터를 읽어왔습니다. (행 개수: {len(df)}개)")
        
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다.")

# 리스트에 담긴 데이터프레임들을 하나로 합치기
if all_df:
    combined_df = pd.concat(all_df, axis=0, ignore_index=True)
    
    # Parquet 파일로 저장
    combined_df.to_parquet(output_parquet, engine='pyarrow', index=False)
    print("-" * 30)
    print(f"최종 병합 완료! 저장 경로: {output_parquet}")
    print(f"전체 데이터 행 개수: {len(combined_df)}개")
else:
    print("합칠 데이터가 없습니다.")