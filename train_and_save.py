import sqlite3
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# DB 연결
conn = sqlite3.connect("data.sqlite")

# 데이터 불러오기
df = pd.read_sql("SELECT * FROM 'ns.lns_reservation';", conn)
df['at_date'] = pd.to_datetime(df['at_date'])
df['weekday'] = df['at_date'].dt.weekday
df = df.drop(columns=['grade', 'ns_link_uid', 'uid'])

# seat 분해
df['seat_row'] = df['seat'].str[0]
df['seat_col'] = df['seat'].str[1:].astype(int)
df['seat_row_num'] = df['seat_row'].apply(lambda x: ord(x.upper()) - ord('A'))

# 학습 데이터 준비
X = df[['weekday', 'at_time', 'seat_row_num', 'seat_col']].astype(float)
le = LabelEncoder()
y = le.fit_transform(df['uuid'])

# 모델 훈련
model = XGBClassifier(tree_method='hist', max_depth=6, n_estimators=100)
model.fit(X, y)

# 모델 저장
model.save_model("trained_model.json")
print("✅ 모델이 'trained_model.json'에 저장되었습니다.")

# LabelEncoder 저장
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
print("✅ 라벨 인코더가 'label_encoder.pkl'로 저장되었습니다.")