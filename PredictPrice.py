#####Importation des bibliothèques
import joblib
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV



#Lire le fichier csv en utilisant pandas
url="https://raw.githubusercontent.com/NGALENAL1004/datasets/master/ford.csv"
dataset = pd.read_csv(url)
#Afficher les données 
print(dataset)
#Suppression de la colonne model
df = dataset.drop(['model'], axis=1)
print(df)


####Analyse du dataset
#Fonction shape
print("Le nombre de lignes et de colonnes est de : ", df.shape)
#Fonction describe
print("Analyse statistique:")
print(df.describe(include = 'all'))
#Le type de variables descriptives
print('Le type de variables:')
print(df.dtypes)
# Sélectionner les variables non numériques
non_numeric_cols = df.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
# Initialiser les listes pour chaque type de variable
binary_cols = []
ordinal_cols = []
nominal_cols = []
# Parcourir les variables non numériques et les classer selon leur type
for col in non_numeric_cols:
    unique_vals = df[col].unique()
    if len(unique_vals) == 2:
        binary_cols.append(col)
    elif df[col].dtype == 'object' or len(unique_vals) <= 5:
        ordinal_cols.append(col)
    else:
        nominal_cols.append(col)
# Afficher les résultats
print('Variable Binaire:', binary_cols)
print('Variable Ordinale:', ordinal_cols)
print('Variable Nominale:', nominal_cols)


###Visualisation des données
#Visualisation des variables catégorielles nominales
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
sns.countplot(x='fuelType', data=df, ax=axs[0])
axs[0].set_title('Distribution des carburants')
axs[0].set_xlabel('Carburant')
axs[0].set_ylabel('Fréquence')
sns.countplot(x='transmission', data=df, ax=axs[1])
axs[1].set_title('Distribution des types de transmissions')
axs[1].set_xlabel('Transmission')
axs[1].set_ylabel('Fréquence')
plt.tight_layout()
plt.show()
#Distribution des années
sns.countplot(x=df.year)
plt.title('Distribution des années')
plt.xlabel('Année')
plt.ylabel('Fréquence')
plt.show()
#Visualisation de distribution et de la moyenne de prix
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
sns.boxenplot(y=df.price, ax=axs[1])
axs[1].set_ylabel('Prix')
axs[1].set_title('Boîtes à moustaches des prix')
sns.histplot(x=df.price, ax=axs[0])
axs[0].set_xlabel('Prix')
axs[0].set_ylabel('Fréquence')
axs[0].set_title('Distribution des prix')
plt.tight_layout()
plt.show()
#Visualisation de la taille des moteurs
fig, axs = plt.subplots(ncols=2, figsize=(20,5))
sns.countplot(x=df.engineSize, ax=axs[0]) 
sns.boxplot(x=df.engineSize,y=df.price,ax=axs[1])
plt.show()
#Visualisation des modeles
fig,axes = plt.subplots(1,2,figsize=(15,10))
sns.countplot(x=dataset.model,ax=axes[0])
axes[0].set_title('Distribution des modèles')
axes[0].tick_params(axis='x',rotation=90)
sns.boxplot(x=dataset.model,y=df.price,ax=axes[1])
axes[1].set_title('Prix en foction des modèles')
axes[1].tick_params(axis='x',rotation=90)
plt.show()
#Les nuages de points (mpg & taxe)
fig, axs = plt.subplots(ncols=2, figsize=(20,5))
sns.scatterplot(x=df.tax, y=df.price, ax=axs[0])
sns.scatterplot(x=df.mpg, y=df.price, ax=axs[1])
fig.suptitle("Nuages de points")
plt.show()
#Les nuages de points (mileage,fueltype,year)
fig, axs = plt.subplots(ncols=3, figsize=(20,5))
sns.scatterplot(x=df.mileage, y=df.price, ax=axs[0])
sns.scatterplot(x=df.fuelType, y=df.price, ax=axs[1])
sns.scatterplot(x=df.year, y=df.price, ax=axs[2])
plt.show()
#Matrice de corrélation
sns.heatmap(df.corr(), annot=True, cmap="RdBu")
plt.show()


####Pré-Processing
#Recherche des valeurs manquantes
print(df.isna().sum())
#Remplacer les valeurs vides par NaN
df.replace("", np.nan, inplace=True)
#Suppression des valeurs abérantes
len = ['price','mileage','tax','mpg']
for i in len:
   x = df[i].describe()
   Q1 = x[4]
   Q3 = x[6]
   IQR = Q3-Q1
   lower_bound = Q1-(1.5*IQR)
   upper_bound = Q3+(1.5*IQR)
   df = df[(df[i]>lower_bound)&(df[i]<upper_bound)]
#Encoder les variables catégorielles
df['transmission'] = df['transmission'].map({'Automatic': 1, 'Manual': 2,'Semi-Auto':3})
df['fuelType'] = df['fuelType'].map({'Petrol': 1, 'Diesel': 2, 'Electric': 3, 'Hybrid':4,'Other':5})
#Le dataset après le pré-processing
print('Le dataset après le pré-processing:')
print(df)


####Apprentissage et test
#Séparation des données en bases d’apprentissage et de test
x = df.drop(['price'], axis=1)
y = df['price']
print(x)
print(y)
# Normaliser les données numériques
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42)
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
# Initialiser les modèles de régression
models = {
    "Gradient Boosting Regressor": GradientBoostingRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "DecisionTreeRegressor":DecisionTreeRegressor()  
}
models_names=["GBR","RFR","LiR","RR","LaR","DTR"]
# Initialisation des listes de résultats pour le training data
mse_training = []
mae_training = []
r2_training = []
mape_training = []
# Boucle pour le trainning data
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    # Calculer les métriques d'évaluation
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    mape = np.mean(np.abs((y_train - y_pred) / y_train)) * 100
    # Ajouter les résultats aux listes
    mse_training.append(mse)
    mae_training.append(mae)
    r2_training.append(r2)
    mape_training.append(mape)
    # Afficher les résultats
    print(name + " Metrics (Training Data):")
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)
    print("R2 Score: ", r2)
    print("Mean Absolute Percentage Error: ", mape)
# Tracer les histogrammes
f, (axe1,axe2,axe3,axe4)=plt.subplots(ncols=4, sharex= True, sharey=False, figsize=(30,10))
axe1.bar(models_names, r2_training)
axe1.set_ylabel('$R^2$')
axe2.bar(models_names, mse_training)
axe2.set_ylabel('MSE')
axe3.bar(models_names, mae_training)
axe3.set_ylabel('MAE')
axe4.bar(models_names, mape_training)
axe4.set_ylabel('MAPE')
plt.show()
# Initialisation des listes de résultats pour le testing data
mse_testing = []
mae_testing = []
r2_testing = []
mape_testing = []
# Boucle pour le testing data
for name, model in models.items():
    y_pred = model.predict(x_test)
    # Calculer les métriques d'évaluation
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    # Ajouter les résultats aux listes
    mse_testing.append(mse)
    mae_testing.append(mae)
    r2_testing.append(r2)
    mape_testing.append(mape)
    # Afficher les résultats
    print(name + " Metrics (Testing Data):")
    print("Mean Squared Error: ", mse)
    print("Mean Absolute Error: ", mae)
    print("R2 Score: ", r2)
    print("Mean Absolute Percentage Error: ", mape)
# Tracer les histogrammes
f, (axe1,axe2,axe3,axe4)=plt.subplots(ncols=4, sharex= True, sharey=False, figsize=(30,10))
axe1.bar(models_names, r2_testing)
axe1.set_ylabel('$R^2$')
axe2.bar(models_names, mse_testing)
axe2.set_ylabel('MSE')
axe3.bar(models_names, mae_testing)
axe3.set_ylabel('MAE')
axe4.bar(models_names, mape_testing)
axe4.set_ylabel('MAPE')
plt.show()
# Instancier le modèle DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
y_predicted = model.predict(x_test)
accuracy=r2_score(y_test, y_pred)
print('le scrore sur le jeu de test est:',accuracy)
#Résultat du modèle
pred=pd.DataFrame.from_dict({'valeur_predite':y_predicted,'valeur_reelle':y_test})
# Apres on doit aujouter le pourcentage d'erreur
pred['difference']=pred.valeur_predite-pred.valeur_reelle
print(pred.sample(n=15).round(2))
print(pred.difference.describe())
#Affichage du résultat du modèle
plt.figure(figsize=(10,8))
plt.title('Valeur réelle vs Valeur prédite',fontsize=25)
sns.scatterplot(x = y_test,y = y_predicted)      
plt.xlabel('Valeur réelle', fontsize=18)                          
plt.ylabel('Valeur prédite', fontsize=16) 
plt.show()


###Optimisation du modèle
##Validation croisée
# Définition des hyperparamètres
parameters = {'max_depth':[None, 3, 5, 7], 
              'min_samples_split':[2, 5, 10], 
              'min_samples_leaf':[1, 2, 4], 
              'max_features':[None, 'sqrt', 'log2']}
# Initialisation du modèle
tree = DecisionTreeRegressor()
# Recherche des meilleurs hyperparamètres par validation croisée
grid_search = GridSearchCV(tree, parameters, cv=5,scoring='r2', n_jobs=-1)
grid_search.fit(x_train, y_train)
# Affichage des meilleurs hyperparamètres et de la performance correspondante
print("Meilleurs hyperparamètres : ")
print(grid_search.best_params_)
print("Meilleure performance (R²) en apprentissage : ")
print(grid_search.best_score_)
# Entraînement du modèle avec les meilleurs hyperparamètres sur l'ensemble d'apprentissage
best_model = DecisionTreeRegressor(max_depth=grid_search.best_params_['max_depth'], 
                                  min_samples_split=grid_search.best_params_['min_samples_split'], 
                                  min_samples_leaf=grid_search.best_params_['min_samples_leaf'], 
                                  max_features=grid_search.best_params_['max_features'])
best_model.fit(x_train, y_train)
# Évaluation du modèle sur l'ensemble de test
y_pred = best_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("Performance (R²) sur l'ensemble de test : ", r2)
#Résultat du modèle avec la validation croisée
pred=pd.DataFrame.from_dict({'valeur_predite':y_pred,'valeur_reelle':y_test})
pred['difference']=pred.valeur_predite-pred.valeur_reelle
print(pred.sample(n=15).round(2))
print(pred.difference.describe())
#Affichage du résultat du modèle avec la validation croisée
plt.figure(figsize=(10,8))
plt.title('Valeur réelle vs Valeur prédite',fontsize=25)
sns.scatterplot(x = y_test,y = y_pred)      
plt.xlabel('Valeur réelle', fontsize=18)                          
plt.ylabel('Valeur prédite', fontsize=16) 
plt.show()


##Variation de la proportion de test
# Boucle sur les valeurs de proportions allant de 10% à 50%
for t in np.arange(0.1, 0.6, 0.1):
    # Séparer les données en données d'apprentissage et de test 
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=t, random_state=42)
    # Entraînement du modèle avec les meilleurs hyperparamètres sur l'ensemble d'apprentissage
    best_model = DecisionTreeRegressor(max_depth=grid_search.best_params_['max_depth'], 
                                    min_samples_split=grid_search.best_params_['min_samples_split'], 
                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'], 
                                    max_features=grid_search.best_params_['max_features'])
    best_model.fit(x_train, y_train)
    # Évaluation du modèle sur l'ensemble de test
    y_pred = best_model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print("Performance (R²) sur l'ensemble de test avec la proportion",t,":", r2)


###Analyse de l'importance de chaque variable
# Séparer les données en données d'apprentissage et de test 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
# Entraînement du modèle avec les meilleurs hyperparamètres sur l'ensemble d'apprentissage
best_model = DecisionTreeRegressor(max_depth=grid_search.best_params_['max_depth'], 
                                    min_samples_split=grid_search.best_params_['min_samples_split'], 
                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'], 
                                    max_features=grid_search.best_params_['max_features'])
best_model.fit(x_train, y_train)
# Évaluation du modèle sur l'ensemble de test
y_pred = best_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("Performance (R²) sur l'ensemble de test :", r2)
# Score d'importance pour chaque variable
importance_scores = best_model.feature_importances_
# Afficher les scores d'importance pour chaque variable
for i, score in enumerate(importance_scores):
    print("Variable {}: Importance Score = {:.2f}".format(i, score))
# Garder les indices des variables avec un score d'importance supérieur à 0.05
indices_to_keep = np.where(importance_scores > 0.05)[0]
columns_to_keep = x_train.columns[indices_to_keep]
print(columns_to_keep)
# Sélection des colonnes pertinentes dans les données 
x_new = x[columns_to_keep]
print(x_new)


###Structure finale du modèle optimisé avec les variables pertinentes
# Séparer les données en données d'apprentissage et de test 
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.4, random_state=42)
# Entraînement du modèle avec les meilleurs hyperparamètres sur l'ensemble d'apprentissage
best_model = DecisionTreeRegressor(max_depth=grid_search.best_params_['max_depth'], 
                                    min_samples_split=grid_search.best_params_['min_samples_split'], 
                                    min_samples_leaf=grid_search.best_params_['min_samples_leaf'], 
                                    max_features=grid_search.best_params_['max_features'])
best_model.fit(x_train, y_train)
# Évaluation du modèle sur l'ensemble de test
y_pred = best_model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("Performance (R²) sur l'ensemble de test :", r2)
#Résultat du modèle avec la validation croisée et la meilleur proportion
pred=pd.DataFrame.from_dict({'valeur_predite':y_pred,'valeur_reelle':y_test})
pred['difference']=pred.valeur_predite-pred.valeur_reelle
print(pred.sample(n=15).round(2))
print(pred.difference.describe())
#Affichage du résultat du modèle avec la validation croisée et la meilleur proportion
plt.figure(figsize=(10,8))
plt.title('Valeur réelle vs Valeur prédite',fontsize=25)
sns.scatterplot(x = y_test,y = y_pred)      
plt.xlabel('Valeur réelle', fontsize=18)                          
plt.ylabel('Valeur prédite', fontsize=16) 
plt.show()
#Prédiction de nouvelles données
new_car = pd.DataFrame({'year': [2022],'mileage': [10000], 'mpg': [40.5], 'engineSize': [1.0]})
y_pred=best_model.predict(new_car)
print("Le modèle prédit un prix de:", int(y_pred))
#Sauvegardage du modèle
joblib.dump(value = best_model, filename = 'predictPriceCar.pkl')
# Utilisation du modèle sauvegarder pour réaliser des prédictions
model_loaded = joblib.load(filename = 'predictPriceCar.pkl')
price=model_loaded.predict(new_car)
print('Le prix est: ',int(price))




