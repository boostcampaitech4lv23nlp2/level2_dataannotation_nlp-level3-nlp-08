import json
import pandas as pd
import numpy as np
from fleiss import fleissKappa

# 정의한 relation 정보 읽어오기
with open("relations.json", 'r', encoding="UTF-8") as j:
     relations = json.loads(j.read())

file_dir = './data/NLP_08_tagtog_result.xlsx'  # 2차 파일럿 태깅

first = pd.read_excel(file_dir, sheet_name='김현수', engine='openpyxl')  
second = pd.read_excel(file_dir, sheet_name='이성구', engine='openpyxl') 
third = pd.read_excel(file_dir, sheet_name='이현준', engine='openpyxl') 
fourth = pd.read_excel(file_dir, sheet_name='조문기', engine='openpyxl') 
fifth = pd.read_excel(file_dir, sheet_name='조익노', engine='openpyxl') 

labels1 = list(first['label'])
labels2 = list(second['label'])
labels3 = list(third['label'])
labels4 = list(fourth['label'])
labels5 = list(fifth['label'])

result = pd.DataFrame()
result['member1'] = [relations[label] for label in labels1]
result['member2'] = [relations[label] for label in labels2]
result['member3'] = [relations[label] for label in labels3]
result['member4'] = [relations[label] for label in labels4]
result['member5'] = [relations[label] for label in labels5]


result = result.to_numpy()
num_classes = int(np.max(result)) + 1  # 라벨이 0부터 시작하기 때문

# 평가자들이 relation별로 예측한 개수 
transformed_result = []
for i in range(len(result)):
    temp = np.zeros(num_classes)
    for j in range(len(result[i])):
        temp[int(result[i][j]-1)] += 1
    transformed_result.append(temp.astype(int).tolist())

# fleiss Kappa 계산
kappa = fleissKappa(transformed_result,len(result[0]))