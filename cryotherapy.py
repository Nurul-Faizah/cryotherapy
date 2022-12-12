#Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import base64
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict
from urllib import parse
from urllib.request import urlopen



#Metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.model_selection import KFold,train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier

github_url = f"https://github.com/Nurul-Faizah/cryotherapy"
UCI_url = f"https://archive.ics.uci.edu/ml/datasets/Cryotherapy+Dataset+"

page_bg_img = """
<style>
[data-testid = "stAppViewContainer"]{
    background-image: url("https://www.mecotec.net/fileadmin/_processed_/2/0/csm_mecotec_cryoairsingle-1600x1066_b534dcdb51.jpg");
    background-size: cover;
}
[data-testid = "stHeader"]{
    background-color:rgba(0,0,0,0);
}
[data-testid = "stAToolbar"]{
    right: 2rem;
}
[data-testid = "stSidebar"]{
    background-image:url("https://images.unsplash.com/photo-1572442132864-fd87d70f3f94?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MjZ8fG5lb24lMjBibHVlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60");
    background-position: center;
}
<style>
"""
st.sidebar.write("""
            # Cryotherapy"""
            )
st.sidebar.write("""
            #### Cryotherapy adalah prosedur pembekuan jaringan di permukaan kulit atau mukosa menggunakan zat yang disebut cryogens. Cryogens yang dipakai dalam prosedur ini meliputi nitrogen cair yang sekarang paling banyak dipakai, carbon dioxide snow, serta dimethyl ether dan propane (DMEP).Prosedur ini dilakukan untuk mengatasi berbagai jenis kelainan kulit seperti kutil dan keloid. Beberapa jenis kanker lain juga bisa dirawat dengan metode ini, antara lain kanker prostat, kanker kanker serviks, kanker tulang, dan kanker hati.Cryotherapy tidak terlalu dianjurkan sebagai pilihan pertama dalam pengobatan untuk Kalangan lanjut usia (lansia) yang tidak bisa menjalani operasi dan Orang yang memiliki lesi kulit yang sudah menyebar luas dan sangat mengganggu penampilan bila dioperasi (misalnya karena akan menyisakan bekas luka)
            """)


st.markdown(page_bg_img,unsafe_allow_html=True)
st.title ("Web Apps - Cryotherapy")
st.write(f"Dataset yang digunakan adalah dataset cryotherapy yang diambil dari situs UCI Mechine Learning Repository [DATASET Cryotherapy]({UCI_url}). Dataset ini memiliki 6 parameter dan 1 parameter untuk 2 class cryotherapy. Pada web ini akan memprediksikan keberhasilan pengobatan penyakit kutil dengan cryotherapy. Anda dapat melihat dataset dan sourcode di repository saya [GITHUB]({github_url}).")

tab_titles = [
    "Accuracy Chart",
    "Implementation",]

tabs = st.tabs(tab_titles)

with tabs[0]:
    df = pd.read_csv('https://raw.githubusercontent.com/Nurul-Faizah/cryotherapy/gh-pages/Cryotherapy.csv', sep=';', quotechar='"')

    # #Data cleaning
    # zero_not_accepted = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    # for column in zero_not_accepted :
    #     df[column] = df[column].replace(0,np.NaN)
    #     mean = int (df[column].mean(skipna=True))
    #     df[column] = df[column].replace(np.NaN,mean)

    #separate target values
    y = df['Result_of_Treatment'].values


    X=df.iloc[:,0:6].values 
    y=df.iloc[:,6].values

    st.write('Jumlah baris dan kolom :', X.shape)
    st.write('Jumlah kelas : ', len(np.unique(y)))

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
    st.write("Data Training :", X_train.shape)
    st.write("Data Testing :", X_test.shape)

    #KNN
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test) 
    accuracy_knn=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_knn = round(knn.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)

    #NAIVE BAYES
    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test) 
    accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for Naive Bayes\n',cm)
    print('accuracy_Naive Bayes: %.3f' %accuracy)
    print('precision_Naive Bayes: %.3f' %precision)
    print('recall_Naive Bayes: %.3f' %recall)
    print('f1-score_Naive Bayes : %.3f' %f1)

    #DECISION TREE
    decision_tree = DecisionTreeClassifier() 
    decision_tree.fit(X_train, y_train)  
    Y_pred = decision_tree.predict(X_test) 
    accuracy_dt=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)

    #ENSEMBLE BAGGING
    ensemble_bagging = BaggingClassifier() 
    ensemble_bagging.fit(X_train, y_train)  
    Y_pred = ensemble_bagging.predict(X_test) 
    accuracy_bg=round(accuracy_score(y_test,Y_pred)* 100, 2)
    acc_ensemble_bagging = round(ensemble_bagging.score(X_train, y_train) * 100, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test,Y_pred)
    precision =precision_score(y_test, Y_pred,average='micro')
    recall =  recall_score(y_test, Y_pred,average='micro')
    f1 = f1_score(y_test,Y_pred,average='micro')
    print('Confusion matrix for DecisionTree\n',cm)
    print('accuracy_DecisionTree: %.3f' %accuracy)
    print('precision_DecisionTree: %.3f' %precision)
    print('recall_DecisionTree: %.3f' %recall)
    print('f1-score_DecisionTree : %.3f' %f1)

    st.write("""
                #### Akurasi:"""
                )

    results = pd.DataFrame({
        'Model': ['K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'],
        'Score': [ acc_knn,acc_gaussian,acc_decision_tree, acc_ensemble_bagging ],
        "Accuracy_score":[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_bg
                        ]})
    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(9)
    st.write(result_df)

    fig = plt.figure()
    fig.patch.set_facecolor('silver')
    fig.patch.set_alpha(0.7)
    ax = fig.add_axes([0,0,1,1])
    ax.patch.set_facecolor('silver')
    ax.patch.set_alpha(0.5)
    ax.plot(['K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'],[accuracy_knn,accuracy_nb,accuracy_dt,accuracy_bg],color='red')
    plt.show()
    st.pyplot(fig)


with tabs[1]:
    with st.expander("Role"):
        (st.write("""
            1. Sex  : Jenis kelamin pasien. 1 = Laki-laki dan 2 = Perempuan.
            2. Age  : Umur pasien.
            3. Time : Waktu berlalu sebelum perawatan.
            4. Number of warts  : Angka kutil
            5. Type : Jenis kutil, 1 = kutil biasa 2 = kutil plantar 3 = kutil lainnya.
            6. Area : Luas permukaan kutil 4 sampai 750 mm^2.
            7. Result of treatment  : Hasil pengobatan, 1 = Pengobatan Berhasil dan 0 = Pengobatan Gagal.
        """))
    col1,col2 = st.columns([2,2])
    model=st.selectbox(
            'Model', ('K-Nearest Neighbor','Naive Bayes','Decision Tree','Ensemble Bagging'))
    with col1:
        a = st.number_input("Sex",0)
        b = st.number_input("Age",0)
        c = st.number_input("Time",0.00)

    with col2:
        d = st.number_input("Number of wart",0)
        e = st.number_input("Type",0)
        f = st.number_input("Area",0)

    submit = st.button('Prediction')

    if submit:
        if model == 'K-Nearest Neighbor':
            X_new = np.array([[a,b,c,d,e,f]])
            predict = knn.predict(X_new)
            if predict == 1 :
                st.write("""# Pengobatan Berhasil""")
            else : 
                st.write("""# Pengobatan Gagal""")

        elif model == 'Naive Bayes':
            X_new = np.array([[a,b,c,d,e,f]])
            predict = gaussian.predict(X_new)
            if predict == 1 :
                st.write("""# Pengobatan Berhasil""")
            else : 
                st.write("""# Pengobatan Gagal""")

        elif model == 'Decision Tree':
            X_new = np.array([[a,b,c,d,e,f]])
            predict = decision_tree.predict(X_new)
            if predict == 1 :
                st.write("""# Pengobatan Berhasil""")
            else : 
                st.write("""# Pengobatan Gagal""")

        else:
            X_new = np.array([[a,b,c,d,e,f]])
            predict = ensemble_bagging.predict(X_new)
            if predict == 1 :
                st.write("""# Pengobatan Berhasil""")
            else : 
                st.write("""# Pengobatan Gagal""")


#deskripsi data berisi penjelasan mengenai parameternya (sudah)
#membuat beberapa model (knn,naive bayes,pohon keputusan,bagging) menggunakan 2 tab untuk model (sudah)
# dan implementasinya menggunakan model dengan akurasi tertinggi (sudah)
#link data, source code ditaruh di github,link github repository (sudah)
#dataset diabetes mellitus (sudah)
        