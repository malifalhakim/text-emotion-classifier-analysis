from backend.classify import classify
import seaborn as sns
import matplotlib.pyplot as plt

def predict_all(df):
    df['emotion'] = df['text'].apply(classify)
    return df

def analyze(df, theme='dark'):
    emotion_counts = df['emotion'].value_counts()
    fig, ax = plt.subplots(facecolor='none')
    
    if theme == 'dark':
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', textprops={'color': 'white'})
    else:
        ax.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%')
    
    return fig
