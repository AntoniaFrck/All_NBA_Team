import argparse
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import leaguedashplayerstats, commonplayerinfo, playerawards
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Sezony brane pod uwagę podczas tworzenia bazy
seasons = ['2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16', '2016-17', '2017-18',
           '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']
seasons_rookie = ['1998-99', '1999-00', '2000-01', '2001-02', '2002-03', '2003-04', '2004-05', '2005-06', '2006-07',
                  '2007-08', '2008-09', '2009-10', '2010-11', '2011-12', '2012-13', '2013-14', '2014-15', '2015-16',
                  '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23']


# Funkcje pomocnicze

#Funkcja do wyświetlania zależności między cechami
def korelacje(df):
    df['AWARDS'].fillna(0, inplace=True)
    df['AWARDS'] = df['AWARDS'].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop(columns=['NICKNAME', 'TEAM_ABBREVIATION', 'POSITION'])
    kolumny_cech = df.columns.drop('PLAYER_NAME')
    korelacje = df[kolumny_cech].corr(method='spearman')
    mask = np.tril(np.ones(korelacje.shape))
    sns.heatmap(korelacje, mask=mask, xticklabels=kolumny_cech[::2],
                yticklabels=kolumny_cech[::2])                     #wyświetlanie heatmapy
    plt.show()
    korelacje_wazne = korelacje[abs(korelacje) > 0.25]             #odfiltrowanie małych korelacji
    korelacje.to_csv(f'korelacje.csv', index=False)                #zapis macierzy korelacji do csv
    korelacje_wazne.to_csv(f'ważne korelacje.csv', index=False)


# Pobranie statystyk i zapis w pliku csv
def pl_stat(year):
    for season in seasons:
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(season=season)
        data = player_stats.get_data_frames()[0]
        data.to_csv('p_s_' + season + '.csv', index=False)


# Przypisanie każdemu zawodnikowi informacje o pozycji i nagrodzie
def position(season):
    player_stats = pd.read_csv(f'stats/p_s_{season}.csv')        #Odczyt danych z pliku csv
    data_df = pd.DataFrame(player_stats)                         # Zamiana danych na DataFrame'a
    data_df['POSITION'] = None                                   #Dodanie pustych kolumn
    data_df['AWARDS'] = None

    #Szukanie pozycji każdego gracza w sezonie
    for player_id in data_df['PLAYER_ID']:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)     #Pobranie CommonPlayerInfo zawodnika
        player_info_data = player_info.get_data_frames()[0]
        player_position = player_info_data['POSITION'].iloc[0]                   #Znalezienie pozycji zawodnika
        index = data_df[data_df['PLAYER_ID'] == player_id].index                 #pobranie indeksu zawodnika
        data_df.loc[index, 'POSITION'] = player_position                         #Przypisanie zawodnikowi znalezionej pozycji
        player_awards = playerawards.PlayerAwards(player_id=player_id)           #Pobranie informacji o nagrodach zawodnika
        awards_data = player_awards.get_data_frames()[0]
        awards_data = pd.DataFrame(awards_data)
        all_nba_awards_season = awards_data[(awards_data['PERSON_ID'] == player_id) &
                                            (awards_data['DESCRIPTION'] == 'All-NBA') &
                                            (awards_data['SEASON'] == season)]
        if not all_nba_awards_season.empty:
            all_nba_team_number = all_nba_awards_season['ALL_NBA_TEAM_NUMBER'].iloc[0]
            data_df.loc[index, 'AWARDS'] = all_nba_team_number                           #Przypisanie zawodnikowi informacji o nagrodzie jeśli ją zdobył

    data_df.to_csv('stats_position_awards' + season + '.csv', index=False)               #Zapis do pliku


# Pobranie informacji o zawodnikach i zapis do pliku csv w zależniści dla każdego sezonu
def com_pl_info(data_df, season):
    com_pl_info = pd.DataFrame()
    for player_id in data_df['PLAYER_ID']:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        player_info_data = player_info.get_data_frames()[0]
        com_pl_info = com_pl_info._append(player_info_data)
    com_pl_info.to_csv('commonplayerinfo' + season + '.csv', index=False)


# Wybranie zawodnikó do nagrody nba all-rookie
def rookie(season):
    cpi = pd.read_csv(f'common player info/commonplayerinfo{season}.csv')           #Odczyt informacji o zawodnikach
    start_year = int(season.split('-')[0])                                          #Uzyskanie początkowego roku sezonu
    cpi_rookie = cpi[cpi['FROM_YEAR'] == start_year].copy()                         #Wybranie zawodników rozpoczynających w tym sezonie
    cpi_rookie = cpi_rookie[['PERSON_ID', 'POSITION']]
    cpi_rookie.to_csv(f'common_pl_info_rookie/comm_pl_inf_rookie_{season}.csv', index=False)


# Pobranie informacji o nagradach dla zawodników rookie
def awards(season):
    data_df = pd.read_csv(f'spa_rookie/spa_rookie_{season}.csv')
    data_df = pd.DataFrame(data_df)
    data_df['AWARDS'] = None
    for player_id in data_df['PLAYER_ID']:
        player_awards = playerawards.PlayerAwards(player_id=player_id)
        awards_data = player_awards.get_data_frames()[0]
        awards_data.to_csv('awards.csv')
        all_nba_awards_season = awards_data[(awards_data['DESCRIPTION'] == 'All-Rookie Team') &
                                            (awards_data['SEASON'] == season)]
        if not all_nba_awards_season.empty:
            all_nba_team_number = all_nba_awards_season['ALL_NBA_TEAM_NUMBER'].iloc[0]
            data_df.loc[data_df['PLAYER_ID'] == player_id, 'AWARDS'] = all_nba_team_number

    data_df.to_csv('spa_rookie_awards/spa_rookie_awards' + season + '.csv', index=False)


# Podział zawodników na pozycje (ostatecznie okazał się zbędny- model działał gorzej gdy brał pod uwagę pozycje)
def spa_position(season):
    spa = pd.read_csv(f'stats_position_awards/stats_position_awards{season}.csv')    #Odczyt danych z pliku
    spa_mp = pd.DataFrame(spa)

    #Filtracja zawodników po pozycjach
    forward_df = spa_mp[(spa_mp['POSITION'] == 'Forward') | (spa_mp['POSITION'] == 'Guard-Forward') | (
            spa_mp['POSITION'] == 'Center-Forward') | (spa_mp['POSITION'] == 'Forward-Guard') | (
                                spa_mp['POSITION'] == 'Forward-Center')]
    center_df = spa_mp[(spa_mp['POSITION'] == 'Center') | (spa_mp['POSITION'] == 'Guard-Center') | (
            spa_mp['POSITION'] == 'Center-Forward') | (spa_mp['POSITION'] == 'Center-Guard') | (
                               spa_mp['POSITION'] == 'Forward-Center')]
    guard_df = spa_mp[(spa_mp['POSITION'] == 'Guard') | (spa_mp['POSITION'] == 'Guard-Center') | (
            spa_mp['POSITION'] == 'Guard-Forward') | (spa_mp['POSITION'] == 'Center-Guard') | (
                              spa_mp['POSITION'] == 'Forward-Guard')]

    #Zapis danych
    center_df.to_csv('spa_center/stats_position_awards_center_' + season + '.csv', index=False)
    forward_df.to_csv('spa_forward/stats_position_awards_forward_' + season + '.csv', index=False)
    guard_df.to_csv('spa_guard/stats_position_awards_guard_' + season + '.csv', index=False)

    return forward_df, center_df, guard_df


# Przygotowanie danych do trenowania modelu
def data(spa):
    spa_df = pd.DataFrame(spa)                #Zamiana danych na DataFrame'a
    spa_df['AWARDS'].fillna(0, inplace=True)  #Wypełnienie pustych pozycji w kolumnie AWARDS zerami
    spa_df['AWARDS'] = spa_df['AWARDS'].apply(lambda
                                                  x: 0 if x == 0 else 1)  # Przypisanie wartości 0 zawodnikom którzy nie znaleźli się w żadnym teamie all-nba i wartości 1 zawodnikom, którzy znaleźli się w którymkolwiek teamie
    spa_df_d = spa_df.drop(columns=['PLAYER_NAME', 'NICKNAME', 'TEAM_ABBREVIATION', 'POSITION'])    #usunięcie zbędnych kolumn
    # label_encoder = LabelEncoder()
    # spa_df_d['POSITION'] = label_encoder.fit_transform(spa_df_d['POSITION'])               #Zamiana wartości POSITION na wartości liczbowe
    y = spa_df_d['AWARDS']
    X = spa_df_d.drop(columns=['AWARDS', 'PLAYER_ID'])

    return X, y, spa_df


# Wybranie ze statystyk tylko tych dotyczących zawodników debiutujących w danym sezonie
def spa_rookie(season):
    spa = pd.read_csv(f'stats_position_awards/stats_position_awards{season}.csv')
    rookie = pd.read_csv(f'common_pl_info_rookie/comm_pl_inf_rookie_{season}.csv')
    spa = pd.DataFrame(spa)
    rookie = pd.DataFrame(rookie)
    spa_rookie = spa[spa['PLAYER_ID'].isin(rookie['PERSON_ID'])]
    spa_rookie.to_csv('spa_rookie/spa_rookie_' + season + '.csv', index=False)

#Wybór teamów
def select_teams(probabilities, test_data):
    top_15 = probabilities.nlargest(15, 1)      #Wybranie 15 zawodników z najwyższym prawdopodobieństwem bycia w ktoryms teamie
    top_15_indices = top_15.index               #Pobranie indeksow zawodnikow
    top_15_names = test_data.loc[top_15_indices, 'PLAYER_NAME']   #Przypisanie nazwisk zawodnikow

    #Podzial na 3 najlepsze piatki
    all_nba_first_team = top_15_names.iloc[:5].tolist()
    all_nba_second_team = top_15_names.iloc[5:10].tolist()
    all_nba_third_team = top_15_names.iloc[10:15].tolist()

    return all_nba_first_team, all_nba_second_team, all_nba_third_team

#Generowanie pliku json z nazwiskami
def generate_json(output_path, first_team, second_team, third_team, first_rookie_team, second_rookie_team):
    output_data = {
        "first all-nba team": first_team,
        "second all-nba team": second_team,
        "third all-nba team": third_team,
        "first rookie all-nba team": first_rookie_team,
        "second rookie all-nba team": second_rookie_team
    }
    with open(output_path, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)


def main(output_path):
    # Wczytywanie danych i trenowanie modeli

    #Utworzenie zbioru z poprzednich sezonow
    spa_f_all = pd.concat(
        [pd.read_csv(f'stats_position_awards/stats_position_awards{season}.csv') for season in seasons],
        ignore_index=True)
    spa_f_r = pd.concat([pd.read_csv(f'spa_rookie_awards/spa_rookie_awards{season}.csv') for season in seasons_rookie],
                        ignore_index=True)

    #Pobranie danych do predykcji
    spa_2_r = pd.read_csv(f'spa_rookie/spa_rookie_2023-24.csv')  # dla nba all-rookie
    spa_2 = pd.read_csv(f'stats_position_awards/stats_position_awards2023-24.csv')  # dla all-nba
    spa_2['AWARDS'] = None

    #Utworzenie zbiorow treningowych i testowych
    X_train, y_train, _ = data(spa_f_all)
    X_test, _, test_table = data(spa_2)

    X_train_r, y_train_r, _ = data(spa_f_r)
    X_test_r, _, test_table_r = data(spa_2_r)

    #Utworzenie i wytrenowanie klasyfikatorow
    clas_r = DecisionTreeClassifier(max_depth=3).fit(X_train_r, y_train_r)  # dla nba all-rookie
    clas = RandomForestClassifier(max_depth=4, random_state=9).fit(X_train, y_train)  # dla all-nba
    joblib.dump(clas_r, 'clas_r_model.pkl')
    joblib.dump(clas, 'clas_model.pkl')

    #Uzyskanie wartosci prawdopodobienstw przynaleznosci do klas
    probabilities = pd.DataFrame(clas.predict_proba(X_test), columns=clas.classes_)
    probabilities_r = pd.DataFrame(clas_r.predict_proba(X_test_r), columns=clas_r.classes_)

    #Wybor teamow
    first_team, second_team, third_team = select_teams(probabilities, test_table)
    first_rookie_team, second_rookie_team, _ = select_teams(probabilities_r, test_table_r)

    #Zapis nazwisk do pliku
    generate_json(output_path, first_team, second_team, third_team, first_rookie_team, second_rookie_team)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NBA.')
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()
    main(args.output_path)
