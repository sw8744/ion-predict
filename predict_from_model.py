import sqlite3
import pandas as pd
import pickle
from xgboost import XGBClassifier

conn = sqlite3.connect("data.sqlite")

df_users = pd.read_sql("SELECT uid, name FROM 'users.users';", conn)
uuid_map = dict(zip(df_users['uid'], df_users['name']))

df = pd.read_sql("SELECT * FROM 'ns.lns_reservation';", conn)
df['at_date'] = pd.to_datetime(df['at_date'])
df['weekday'] = df['at_date'].dt.weekday
df['seat_row'] = df['seat'].str[0]
df['seat_col'] = df['seat'].str[1:].astype(int)
df['seat_row_num'] = df['seat_row'].apply(lambda x: ord(x.upper()) - ord('A'))

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

model = XGBClassifier()
model.load_model("trained_model.json")

encoded_index_to_uuid = {i: le.classes_[i] for i in range(len(le.classes_))}

def predict_uuid(weekday, at_time, seat):
    seat_row = seat[0]
    seat_col = int(seat[1:])
    seat_row_num = ord(seat_row.upper()) - ord('A')
    X_input = pd.DataFrame([[weekday, at_time, seat_row_num, seat_col]],
                           columns=['weekday', 'at_time', 'seat_row_num', 'seat_col'])
    pred_label = model.predict(X_input)[0]
    return encoded_index_to_uuid.get(pred_label, "Unknown")

def uuid_to_name(uuid):
    return uuid_map.get(uuid, "Unknown")

def list_students_for_seat(weekday, at_time, seat):
    seat = seat.strip().upper()
    matched = df[
        (df['weekday'] == weekday) &
        (df['at_time'] == str(at_time)) &
        (df['seat'] == seat)
    ]
    return [uuid_map.get(u, u) for u in matched['uuid'].unique()]

def most_frequent_students_for_seat(weekday, at_time, seat):
    seat = seat.strip().upper()
    matched = df[
        (df['weekday'] == weekday) &
        (df['at_time'] == str(at_time)) &
        (df['seat'] == seat)
    ]
    counts = matched['uuid'].value_counts()
    return [(uuid_map.get(uid, uid), count) for uid, count in counts.items()]

# 사용 예시
if __name__ == "__main__":
    w, t, s = 0, 2, "E2" # w -> 0부터 순서대로 월 화 수 목 금
    predicted_uuid = predict_uuid(w, t, s)
    predicted_name = uuid_to_name(predicted_uuid)
    print("예측된 학생:", predicted_name)
    print("실제 해당 자리에 앉았던 학생들:", list_students_for_seat(w, t, s))
    print("가장 자주 앉은 학생:")
    for name, count in most_frequent_students_for_seat(w, t, s):
        print(f" - {name}: {count}회")