import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

crimes = pd.read_csv('crimes2018.csv',sep=',',low_memory=False)
crimes = crimes[:10000]
crimes = crimes.drop('ID',axis = 1)
crimes = crimes.drop('Case Number',axis = 1)
crimes = crimes.drop('District',axis = 1)
crimes = crimes.drop('Ward',axis = 1)
crimes = crimes.drop('X Coordinate',axis = 1)
crimes = crimes.drop('Y Coordinate',axis = 1)
crimes = crimes.drop('Updated On',axis = 1)
crimes = crimes.drop('Longitude',axis = 1)
crimes = crimes.drop('Latitude',axis = 1)
crimes = crimes.drop('Location',axis = 1)
crimes = crimes.drop('Block',axis = 1)
crimes = crimes.drop('IUCR',axis = 1)
crimes = crimes.drop('Description',axis = 1)
crimes = crimes.drop('Beat',axis = 1)
crimes = crimes.drop('Community Area',axis = 1)
crimes = crimes.drop('Primary Type',axis = 1)
crimes = crimes.drop('Year',axis = 1)

crimes = crimes.dropna()

Dict = {'APARTMENT': 'RESIDENTIAL',
        'CHA APARTMENT': 'RESIDENTIAL',
        'CTA TRAIN': 'TRANSPORT',
        'RESIDENCE': 'RESIDENTIAL',
        'BANK': 'COMMERCIAL',
        'VEHICLE NON-COMMERCIAL': 'TRANSPORT',
        'STREET': 'TRANSPORT',
        'OTHER': 'OTHER',
        'BAR OR TAVERN': 'COMMERCIAL',
        'PARKING LOT/GARAGE(NON.RESID.)': 'TRANSPORT',
        'SIDEWALK': 'OTHER',
        'BARBERSHOP': 'COMMERCIAL',
        'POLICE FACILITY/VEH PARKING LOT': 'GOVERNMENT',
        'COMMERCIAL / BUSINESS OFFICE': 'COMMERCIAL',
        'CURRENCY EXCHANGE': 'COMMERCIAL',
        'AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA': 'AIRPORT',
        'HOSPITAL BUILDING/GROUNDS': 'GOVERNMENT',
        'DEPARTMENT STORE': 'COMMERCIAL',
        'AUTO': 'COMMERCIAL',
        'PARK PROPERTY': 'OTHER',
        'HOTEL/MOTEL': 'COMMERCIAL',
        'RESTAURANT': 'COMMERCIAL',
        'ATHLETIC CLUB': 'COMMERCIAL',
        'ALLEY': 'OTHER',
        'NURSING HOME/RETIREMENT HOME': 'RESIDENTIAL',
        'SMALL RETAIL STORE': 'COMMERCIAL',
        'RESIDENCE-GARAGE': 'RESIDENTIAL',
        'VEHICLE - OTHER RIDE SHARE SERVICE (E.G., UBER, LYFT)': 'TRANSPORT',
        'SCHOOL, PUBLIC, BUILDING': 'EDUCATION',
        'CHURCH/SYNAGOGUE/PLACE OF WORSHIP': 'OTHER',
        'AUTO / BOAT / RV DEALERSHIP': 'COMMERCIAL',
        'MOVIE HOUSE/THEATER': 'COMMERCIAL',
        'CTA PLATFORM': 'TRANSPORT',
        'CEMETARY': 'OTHER',
        'SCHOOL, PRIVATE, BUILDING': 'EDUCATION',
        'GAS STATION': 'COMMERCIAL',
        'RESIDENCE PORCH/HALLWAY': 'RESIDENTIAL',
        'GOVERNMENT BUILDING/PROPERTY': 'GOVERNMENT',
        'VACANT LOT/LAND': 'OTHER',
        'GROCERY FOOD STORE' : 'COMMERCIAL',
        'RESIDENTIAL YARD (FRONT/BACK)': 'RESIDENTIAL',
        'TAVERN/LIQUOR STORE': 'COMMERCIAL',
        'CTA STATION' : 'TRANSPORT',
        'WAREHOUSE': 'COMMERCIAL',
        'AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA': 'AIRPORT',
        'SCHOOL, PUBLIC, GROUNDS': 'EDUCATION',
        'OTHER RAILROAD PROP / TRAIN DEPOT': 'TRANSPORT',
        'AIRPORT/AIRCRAFT': 'AIRPORT',
        'MEDICAL/DENTAL OFFICE': 'COMMERCIAL',
        'CONVENIENCE STORE': 'COMMERCIAL',
        'TAXICAB': 'TRANSPORT',
        'ABANDONED BUILDING': 'OTHER',
        'CTA BUS STOP': 'TRANSPORT',
        'ATM (AUTOMATIC TELLER MACHINE)': 'COMMERCIAL',
        'AIRPORT TERMINAL UPPER LEVEL - SECURE AREA' : 'AIRPORT',
        'POOL ROOM': 'RESIDENTIAL',
        'AIRPORT VENDING ESTABLISHMENT': 'AIRPORT',
        'DAY CARE CENTER': 'COMMERCIAL',
        'VEHICLE-COMMERCIAL': 'COMMERCIAL',
        'DRUG STORE': 'COMMERCIAL',
        'CHA HALLWAY/STAIRWELL/ELEVATOR': 'GOVERNMENT',
        'AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA': 'AIRPORT',
        'CTA BUS': 'TRANSPORT',
        'CONSTRUCTION SITE': 'COMMERCIAL',
        'FACTORY/MANUFACTURING BUILDING': 'COMMERCIAL',
        'CTA GARAGE / OTHER PROPERTY': 'TRANSPORT', 
        'COLLEGE/UNIVERSITY GROUNDS': 'EDUCATION',
        'CAR WASH': 'COMMERCIAL',
        'JAIL / LOCK-UP FACILITY' : 'GOVERNMENT',
        'AIRCRAFT': 'AIRPORT',
        'LIBRARY' : 'COMMERCIAL',
        'DRIVEWAY - RESIDENTIAL': 'RESIDENTIAL',
        'AIRPORT BUILDING NON-TERMINAL - SECURE AREA': 'AIRPORT',
        'VEHICLE - DELIVERY TRUCK': 'TRANSPORT',
        'AIRPORT EXTERIOR - SECURE AREA': 'AIRPORT',
        'AIRPORT TERMINAL LOWER LEVEL - SECURE AREA': 'AIRPORT',
        'LAKEFRONT/WATERFRONT/RIVERBANK': 'OTHER',
        'AIRPORT PARKING LOT': 'AIRPORT',
        'AIRPORT EXTERIOR - NON-SECURE AREA': 'AIRPORT',
        'SCHOOL, PRIVATE, GROUNDS': 'EDUCATION',
        'SPORTS ARENA/STADIUM': 'COMMERCIAL',
        'AIRPORT TRANSPORTATION SYSTEM (ATS)': 'AIRPORT',
        'AIRPORT TERMINAL MEZZANINE - NON-SECURE AREA': 'AIRPORT',
        'APPLIANCE STORE': 'COMMERCIAL',
        'OTHER COMMERCIAL TRANSPORTATION': 'TRANSPORT',
        'COIN OPERATED MACHINE': 'COMMERCIAL',
        'ANIMAL HOSPITAL': 'GOVERNMENT',
        'CREDIT UNION': 'COMMERCIAL',
        'SAVINGS AND LOAN': 'COMMERCIAL',
        'HIGHWAY/EXPRESSWAY': 'TRANSPORT',
        'CHA PARKING LOT/GROUNDS': 'GOVERNMENT',
        'CLEANING STORE': 'COMMERCIAL', 
        'PAWN SHOP': 'COMMERCIAL',
        'BRIDGE': 'TRANSPORT',
        'COLLEGE/UNIVERSITY RESIDENCE HALL': 'EDUCATION',
        'FOREST PRESERVE': 'OTHER',
        'FEDERAL BUILDING': 'GOVERNMENT',
        'NEWSSTAND': 'COMMERCIAL',
        'BOWLING ALLEY': 'COMMERCIAL',
        'VEHICLE-COMMERCIAL - TROLLEY BUS': 'TRANSPORT',
        'FIRE STATION': 'GOVERNMENT',
        'CTA TRACKS - RIGHT OF WAY' : 'TRANSPORT',
        'BOAT/WATERCRAFT': 'TRANSPORT',
        'VEHICLE-COMMERCIAL - ENTERTAINMENT/PARTY BUS': 'COMMERCIAL',
        'HORSE STABLE': 'RESIDENTIAL',
        'FARM': 'RESIDENTIAL',
        'KENNEL': 'COMMERCIAL',
        'PARKING LOT': 'TRANSPORT',
        'HOUSE': 'RESIDENTIAL',
        'CTA PROPERTY' : 'TRANSPORT',
        'RETAIL STORE': 'COMMERCIAL',
        'HOTEL': 'COMMERCIAL',
        'PORCH': 'OTHER',
        'VACANT LOT' : 'TRANSPORT',
        'YMCA': 'COMMERCIAL',
        'GOVERNMENT BUILDING': 'GOVERNMENT',
        'DRIVEWAY': 'TRANSPORT',
        'GARAGE/AUTO REPAIR': 'COMMERCIAL',
        'HALLWAY': 'RESIDENTIAL',
        'YARD': 'RESIDENTIAL',
        'CHA GROUNDS' : 'GOVERNMENT',
        'GARAGE': 'TRANSPORT',
        'LIQUOR STORE': 'COMMERCIAL',
        'RIVER BANK': 'OTHER',
        'GAS STATION DRIVE/PROP.'
        'WOODED AREA': 'OTHER',
        'OFFICE': 'COMMERCIAL',
        'BARBER SHOP/BEAUTY SALON': 'COMMERCIAL',
        'STAIRWELL': 'OTHER',
        'HOSPITAL': 'GOVERNMENT',
        'CHA PARKING LOT' : 'GOVERNMENT',
        'GANGWAY': 'OTHER'}

for index, row in crimes.iterrows():
    crimes.loc[index, 'Location Description'] = Dict[crimes.loc[index, 'Location Description']];

le = LabelEncoder()
        
X = crimes.drop('Location Description', axis = 1)
y = crimes['Location Description']
def dummyEncode(df):
          columnsToEncode = list(df.select_dtypes(include=['category','object','bool']))
          le = LabelEncoder()
          for feature in columnsToEncode:
              try:
                  df[feature] = le.fit_transform(df[feature])
              except:
                  print('Error encoding '+feature)
          return df
    
X = dummyEncode(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Neural Network
mlpc = MLPClassifier(hidden_layer_sizes = (11,11,11), max_iter = 1000)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)

#Random Forest
rfc = RandomForestClassifier(n_estimators = 100, random_state = 42)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

#Lets see how our model preformed
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))
print(accuracy_score(y_test, pred_mlpc))
sns.countplot(pred_mlpc)

#Export Model
filenameOne = "neuralnet.sav"
filenameTwo = "randomforest.sav"
filenameThree = "scaler.sav"

pickle.dump(mlpc, open(filenameOne, 'wb'))
pickle.dump(rfc, open(filenameTwo, 'wb'))
pickle.dump(sc, open(filenameThree, 'wb'))