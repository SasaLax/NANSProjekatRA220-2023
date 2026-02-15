from project_helper import *

def main():
    path = 'PRSA_Data_Aotizhongxin_20130301-20170228.csv'

    try:
        df = load_air_quality_data(path)
        df = preprocess_missing_values(df)
        print(f"Podaci uspjesno ucitani. Ukupan broj uzoraka: {len(df)}") #35064

        run_eda_correlation(df) #PM2.5 ima jasnu negativnu korelaciju sa brzinom vjetra i temp
        run_stl_decomposition(df) #Potvrdjena jasna sezonalnost

        # if df['PM2.5'].isnull().sum() == 0:
        #     print("Nema nedostajucih vrijednosti.")

        # print("\nPregled obradjenih podataka:")
        # print(df.head())
    
    except FileNotFoundError:
        print(f"Greska: Fajl nije pronadjen.")
    except Exception as e:
        print(f"Doslo je do neocekivane greske: {e}")

if __name__ == "__main__":
    main()