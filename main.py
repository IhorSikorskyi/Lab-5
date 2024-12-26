import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def preprocess_data(file_path):
    # Завантаження даних
    data = pd.read_csv(file_path)

    # Аналіз пропущених значень
    print("Пропущені значення:\n", data.isnull().sum())

    # Заповнення пропущених числових значень медіаною
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[num_columns] = data[num_columns].fillna(data[num_columns].median())

    # Заповнення пропущених категоріальних значень модою
    cat_columns = data.select_dtypes(include=['object']).columns
    data[cat_columns] = data[cat_columns].fillna(data[cat_columns].mode().iloc[0])

    # Перетворення категоріальних змінних на числові значення
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    print("Пропущені значення:\n", data.isnull().sum())

    print("\nПерші рядки оброблених даних:\n", data.head())
    return data

file_path = "heart_disease_uci.csv"

# Попередня обробка даних
data = preprocess_data(file_path)

# Розділення даних на X і y
X = data.drop(columns=['num'])
y = data['num']

individual = [128, 128, 64, 64, 32, 'relu', 'relu', 'relu', 'relu', 'relu']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_layers = len(individual) // 2
neurons = individual[:num_layers]
activations = individual[num_layers:]
# Build and train the neural network model
model = Sequential()
for i in range(num_layers):
    model.add(Dense(units=neurons[i], activation=activations[i]))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

_, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Точність (нейронна мережа): {accuracy:.2f}\n")

def rescale_for_fuzzy(input_value, min_val, max_val):
    return input_value * (max_val - min_val) + min_val

def fuzzy_system(age_value, sex_value, cp_value, blood_pressure_value, cholesterol_value,
                 fbs_value, restecg_value, thalach_value, exang_value, oldpeak_value,
                 slope_value, ca_value, thal_value, nn_risk_value):
    # Створення нечітких змінних
    age = ctrl.Antecedent(np.arange(0, 101, 1), 'Age')
    sex = ctrl.Antecedent(np.arange(0, 2, 1), 'Sex')  # 0 - Жінка, 1 - Чоловік
    cp = ctrl.Antecedent(np.arange(0, 4, 1), 'CP')  # Тип болю в грудях
    blood_pressure = ctrl.Antecedent(np.arange(50, 201, 1), 'BloodPressure')
    cholesterol = ctrl.Antecedent(np.arange(0, 301, 1), 'Cholesterol')
    fbs = ctrl.Antecedent(np.arange(0, 2, 1), 'FBS')  # Фастингова кров
    restecg = ctrl.Antecedent(np.arange(0, 3, 1), 'RestECG')  # Результати ЕКГ
    thalch = ctrl.Antecedent(np.arange(50, 201, 1), 'Thalch')  # Максимальна ЧСС
    exang = ctrl.Antecedent(np.arange(0, 2, 1), 'Exang')  # Ангіна при навантаженні
    oldpeak = ctrl.Antecedent(np.arange(0, 6, 0.1), 'OldPeak')  # Депресія ST при навантаженні
    slope = ctrl.Antecedent(np.arange(0, 4, 1), 'Slope')  # Схил сегменту ST
    ca = ctrl.Antecedent(np.arange(0, 4, 1), 'CA')  # Кількість судин
    thal = ctrl.Antecedent(np.arange(0, 4, 1), 'Thal')  # Тип дефекту
    nn_risk = ctrl.Antecedent(np.arange(0, 1, 1.1), 'NNRisk')
    risk = ctrl.Consequent(np.arange(0, 11, 1), 'Risk')

    # Визначення нечітких функцій належності
    # Вік
    age['young'] = fuzz.trimf(age.universe, [0, 25, 35])
    age['middle_aged'] = fuzz.trimf(age.universe, [30, 50, 70])
    age['old'] = fuzz.trimf(age.universe, [60, 80, 100])

    # Холестерин
    cholesterol['low'] = fuzz.trimf(cholesterol.universe, [0, 100, 200])
    cholesterol['normal'] = fuzz.trimf(cholesterol.universe, [100, 150, 210])
    cholesterol['high'] = fuzz.trimf(cholesterol.universe, [200, 450, 610])

    # Артеріальний тиск
    blood_pressure['low'] = fuzz.trimf(blood_pressure.universe, [50, 70, 90])
    blood_pressure['normal'] = fuzz.trimf(blood_pressure.universe, [80, 110, 140])
    blood_pressure['high'] = fuzz.trimf(blood_pressure.universe, [130, 160, 200])

    # Стать
    sex['female'] = fuzz.trimf(sex.universe, [0, 0, 1])
    sex['male'] = fuzz.trimf(sex.universe, [0, 1, 1])

    # Тип болю в грудях
    cp['typical_angina'] = fuzz.trimf(cp.universe, [0, 1, 2])
    cp['atypical_angina'] = fuzz.trimf(cp.universe, [1, 2, 3])
    cp['non_anginal'] = fuzz.trimf(cp.universe, [2, 3, 3])
    cp['asymptomatic'] = fuzz.trimf(cp.universe, [0, 0, 1])

    # Фастингова кров
    fbs['low'] = fuzz.trimf(fbs.universe, [0, 0, 1])
    fbs['high'] = fuzz.trimf(fbs.universe, [0, 1, 1])

    # Результати ЕКГ
    restecg['normal'] = fuzz.trimf(restecg.universe, [0, 0, 1])
    restecg['stt_abnormality'] = fuzz.trimf(restecg.universe, [0, 1, 2])
    restecg['lv_hypertrophy'] = fuzz.trimf(restecg.universe, [1, 2, 2])

    # Максимальна ЧСС
    thalch['low'] = fuzz.trimf(thalch.universe, [50, 80, 120])
    thalch['normal'] = fuzz.trimf(thalch.universe, [100, 150, 170])
    thalch['high'] = fuzz.trimf(thalch.universe, [160, 180, 210])

    # Ангіна при навантаженні
    exang['no'] = fuzz.trimf(exang.universe, [0, 0, 1])
    exang['yes'] = fuzz.trimf(exang.universe, [0, 1, 1])

    # Депресія ST при навантаженні
    oldpeak['low'] = fuzz.trimf(oldpeak.universe, [-3, 0, 2])
    oldpeak['moderate'] = fuzz.trimf(oldpeak.universe, [1, 2, 3])
    oldpeak['high'] = fuzz.trimf(oldpeak.universe, [3, 4, 5])

    # Схил сегменту ST
    slope['up'] = fuzz.trimf(slope.universe, [0, 0, 1])
    slope['flat'] = fuzz.trimf(slope.universe, [1, 1, 2])
    slope['down'] = fuzz.trimf(slope.universe, [1, 2, 2])

    # Кількість судин
    ca['low'] = fuzz.trimf(ca.universe, [0, 0, 1])
    ca['moderate'] = fuzz.trimf(ca.universe, [1, 2, 2])
    ca['high'] = fuzz.trimf(ca.universe, [2, 3, 3])

    # Тип дефекту
    thal['normal'] = fuzz.trimf(thal.universe, [0, 0, 1])
    thal['fixed_defect'] = fuzz.trimf(thal.universe, [0, 1, 2])
    thal['reversible_defect'] = fuzz.trimf(thal.universe, [1, 2, 2])

    # Ризик_NN
    nn_risk['low'] = fuzz.trimf(nn_risk.universe, [0, 0.1, 0.4])
    nn_risk['medium'] = fuzz.trimf(nn_risk.universe, [0.3, 0.5, 0.7])
    nn_risk['high'] = fuzz.trimf(nn_risk.universe, [0.7, 0.9, 1.0])

    # Ризик
    risk['low'] = fuzz.trimf(risk.universe, [0, 1, 4])
    risk['medium'] = fuzz.trimf(risk.universe, [3, 5, 7])
    risk['high'] = fuzz.trimf(risk.universe, [7, 9, 10])

    # Створення нечітких правил
    rule1 = ctrl.Rule(age['young'] & cholesterol['low'] & blood_pressure['low'], risk['medium'])
    rule2 = ctrl.Rule(age['young'] & cholesterol['normal'] & blood_pressure['normal'], risk['low'])
    rule3 = ctrl.Rule(age['young'] & cholesterol['high'] & blood_pressure['high'], risk['medium'])
    rule4 = ctrl.Rule(age['young'] & cp['asymptomatic'], risk['low'])
    rule5 = ctrl.Rule(age['young'] & cp['typical_angina'] & thalch['high'], risk['medium'])
    rule6 = ctrl.Rule(age['young'] & cp['atypical_angina'] & thalch['high'], risk['high'])
    rule7 = ctrl.Rule(age['young'] & cholesterol['high'] & thalch['high'] & restecg['normal'], risk['high'])
    rule8 = ctrl.Rule(age['young'] & cholesterol['high'] & blood_pressure['high'], risk['medium'])

    rule9 = ctrl.Rule(age['middle_aged'] & cholesterol['low'] & blood_pressure['normal'] & sex['male'], risk['medium'])
    rule10 = ctrl.Rule(age['middle_aged'] & cholesterol['low'] & blood_pressure['normal'] & sex['female'], risk['low'])
    rule11 = ctrl.Rule(age['middle_aged'] & cholesterol['normal'] & blood_pressure['normal'], risk['low'])
    rule12 = ctrl.Rule(age['middle_aged'] & cholesterol['high'] & blood_pressure['high'], risk['high'])
    rule13 = ctrl.Rule(age['middle_aged'] & cholesterol['normal'] & blood_pressure['high'],risk['medium'])
    rule14 = ctrl.Rule(age['middle_aged'] & cholesterol['high'] & blood_pressure['normal'],risk['medium'])
    rule15 = ctrl.Rule(age['middle_aged'] & cholesterol['low'] & blood_pressure['low'], risk['medium'])
    rule16 = ctrl.Rule(age['middle_aged'] & cholesterol['high'] & thalch['normal'] & exang['yes'], risk['high'])
    rule17 = ctrl.Rule(age['middle_aged'] & cp['atypical_angina'] & thalch['low'], risk['medium'])
    rule18 = ctrl.Rule(age['middle_aged'] & cholesterol['high'] & blood_pressure['low'], risk['medium'])
    rule19 = ctrl.Rule(age['middle_aged'] & cholesterol['low'] & blood_pressure['normal'], risk['low'])

    rule20 = ctrl.Rule(age['old'] & cholesterol['low'] & blood_pressure['high'], risk['medium'])
    rule21 = ctrl.Rule(age['old'] & cholesterol['normal'] & blood_pressure['high'], risk['high'])
    rule22 = ctrl.Rule(age['old'] & cholesterol['high'] & blood_pressure['high'], risk['high'])
    rule23 = ctrl.Rule(age['old'] & cholesterol['normal'] & blood_pressure['high'], risk['high'])
    rule24 = ctrl.Rule(age['old'] & cholesterol['normal'] & blood_pressure['normal'], risk['medium'])
    rule25 = ctrl.Rule(age['old'] & cholesterol['low'] & blood_pressure['high'], risk['medium'])
    rule26 = ctrl.Rule(age['old'] & oldpeak['high'] & thalch['low'], risk['high'])

    rule27 = ctrl.Rule(cp['typical_angina'] & thalch['low'] & oldpeak['high'], risk['high'])
    rule28 = ctrl.Rule(exang['yes'] & restecg['lv_hypertrophy'] & thal['reversible_defect'], risk['medium'])
    rule29 = ctrl.Rule(ca['high'] & oldpeak['moderate'], risk['medium'])
    rule30 = ctrl.Rule(sex['female'] & cp['non_anginal'], risk['low'])
    rule31 = ctrl.Rule(fbs['high'] & blood_pressure['high'], risk['high'])
    rule32 = ctrl.Rule(slope['down'] & oldpeak['high'], risk['high'])
    rule33 = ctrl.Rule(thalch['normal'] & restecg['normal'] & exang['no'], risk['low'])
    rule34 = ctrl.Rule(sex['male'] & cp['typical_angina'] & oldpeak['high'], risk['medium'])
    rule35 = ctrl.Rule(sex['female'] & cp['typical_angina'] & oldpeak['high'], risk['low'])
    rule36 = ctrl.Rule(sex['female'] & cholesterol['high'] & blood_pressure['high'], risk['medium'])
    rule37 = ctrl.Rule(sex['female'] & cholesterol['low'] & blood_pressure['normal'], risk['low'])

    rule38 = ctrl.Rule(nn_risk['high'] & cholesterol['low'] & blood_pressure['normal'] & thalch['normal'], risk['medium'])
    rule39 = ctrl.Rule(nn_risk['medium'] & cholesterol['high'] & blood_pressure['high'], risk['high'])
    rule40 = ctrl.Rule(nn_risk['low'] & oldpeak['high'] & thalch['low'], risk['high'])
    rule41 = ctrl.Rule(nn_risk['medium'] & cp['typical_angina'] & slope['down'], risk['medium'])
    rule42 = ctrl.Rule(nn_risk['high'] & cp['non_anginal'] & blood_pressure['low'], risk['medium'])
    rule43 = ctrl.Rule(nn_risk['low'] & cholesterol['low'] & blood_pressure['normal'], risk['low'])

    # Створення контролера для системи
    rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10,
             rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20,
             rule21, rule22, rule23, rule24, rule25, rule26, rule27, rule28, rule29, rule30,
             rule31, rule32, rule33, rule34, rule35, rule36, rule37, rule38, rule39, rule40,
             rule41, rule42, rule43]
    risk_ctrl = ctrl.ControlSystem(rules)
    risk_decision = ctrl.ControlSystemSimulation(risk_ctrl)

    # Обчислення результату
    risk_decision.input['Age'] = age_value
    risk_decision.input['Sex'] = 1 if sex_value == 'male' else 0
    risk_decision.input['CP'] = cp_value
    risk_decision.input['BloodPressure'] = blood_pressure_value
    risk_decision.input['Cholesterol'] = cholesterol_value
    risk_decision.input['FBS'] = 1 if fbs_value else 0
    risk_decision.input['RestECG'] = restecg_value
    risk_decision.input['Thalch'] = thalach_value
    risk_decision.input['Exang'] = 1 if exang_value else 0
    risk_decision.input['OldPeak'] = oldpeak_value
    risk_decision.input['Slope'] = slope_value
    risk_decision.input['CA'] = ca_value
    risk_decision.input['Thal'] = thal_value
    risk_decision.input['NNRisk'] = nn_risk_value

    # Обчислення результату
    risk_decision.compute()
    return risk_decision.output['Risk']

def visualize_nn_vs_fuzzy(nn_predictions, fuzzy_predictions, true_labels):
    indices = np.arange(len(nn_predictions))
    width = 0.3

    plt.figure(figsize=(12, 6))
    plt.bar(indices - width, nn_predictions, width, label='NN Predictions', color='blue')
    plt.bar(indices, fuzzy_predictions, width, label='Fuzzy Predictions', color='green')
    plt.bar(indices + width, true_labels, width, label='True Labels', color='red')

    plt.xlabel("Samples")
    plt.ylabel("Risk Level")
    plt.title("Comparison of NN and Fuzzy Logic Predictions")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Генерація випадкових значень для параметрів
    nn_predictions = []
    fuzzy_predictions = []
    for i in range(6):
        age = X_test.iloc[i]['age']
        sex = X_test.iloc[i]['sex']
        cp = X_test.iloc[i]['cp']
        trestbps = X_test.iloc[i]['trestbps']
        chol = X_test.iloc[i]['chol']
        fbs = X_test.iloc[i]['fbs']
        restecg = X_test.iloc[i]['restecg']
        thalch = X_test.iloc[i]['thalch']
        exang = X_test.iloc[i]['exang']
        oldpeak = X_test.iloc[i]['oldpeak']
        slope = X_test.iloc[i]['slope']
        ca = X_test.iloc[i]['ca']
        thal = X_test.iloc[i]['thal']

        prediction = model.predict(X_test.iloc[[i]])
        nn_risk = np.mean(prediction)
        print(f"Нейронна мережа оцінює ризик: {nn_risk:.2f}")

        # Оцінка ризику за допомогою нечіткої системи
        risk = fuzzy_system(age, sex, cp, trestbps,
                            chol, fbs, restecg, thalch,
                            exang, oldpeak, slope, ca, thal, nn_risk)
        print(f"Ризик для пацієнта №{i+1} (нечітка система): {risk:.2f} з 10")

        nn_predictions.append(nn_risk)
        fuzzy_predictions.append(risk)
    true_labels = y_test[:len(nn_predictions)]
    visualize_nn_vs_fuzzy(nn_predictions, fuzzy_predictions,true_labels)
