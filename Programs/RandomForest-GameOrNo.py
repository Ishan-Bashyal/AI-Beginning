from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rain','Rain','Rain','Overcast','Sunny',
                'Sunny','Rain','Sunny','Overcast','Overcast','Rain'],
    'Temp': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
             'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Wind': ['Weak','Strong','Weak','Weak','Weak','Strong','Strong','Weak',
             'Weak','Weak','Strong','Strong','Weak','Strong'],
    'Decision': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

feature_encoders = {}
for column in ['Outlook','Temp','Humidity','Wind']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    feature_encoders[column] = le  # store encoder for later use

target_encoder = LabelEncoder()
df['Decision'] = target_encoder.fit_transform(df['Decision'])

X = df[['Outlook','Temp','Humidity','Wind']]
y = df['Decision']

clf = DecisionTreeClassifier(criterion='entropy')  # ID3 algorithm
clf.fit(X, y)

tree_rules = export_text(clf, feature_names=list(X.columns))
print(tree_rules)

new_day = pd.DataFrame({
    'Outlook':[feature_encoders['Outlook'].transform(['Sunny'])[0]],
    'Temp':[feature_encoders['Temp'].transform(['Hot'])[0]],
    'Humidity':[feature_encoders['Humidity'].transform(['High'])[0]],
    'Wind':[feature_encoders['Wind'].transform(['Weak'])[0]]
})

prediction = clf.predict(new_day)
prediction_label = target_encoder.inverse_transform(prediction)
print("Decision for new day:", prediction_label[0])
