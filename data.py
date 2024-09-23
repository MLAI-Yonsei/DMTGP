import os
import copy 

import numpy as np
import pandas as pd
import datetime as dt

import paths

# preprocess raw data
def preprocess_data(raw_root, prp_root, start_date, pred_end_date):
    start_date = np.datetime64(start_date)
    pred_end_date = np.datetime64(pred_end_date)
    date_range = pd.date_range(start_date, pred_end_date)
    num_days = int((pred_end_date - start_date) / np.timedelta64(1, 'D')) + 1 

    nation_dict = {
        'uk': ['United Kingdom', '영국'],
        'japan': ['Japan', '일본'],
        'taiwan': ['Taiwan', '대만'],
        'south_korea': ['South Korea', '한국'],
        'france': ['France', '프랑스'],
        'germany': ['Germany', '독일'],
        'denmark': ['Denmark', '덴마크'],
        'italy': ['Italy', '이탈리아'],
        'us': ['US', '미국'],
        'uk': ['United Kingdom', '영국'],
    }
    
    # COVID-19 variant info
    variants_vocab = {}
    for key in nation_dict.keys():
        nation = nation_dict[key][0]
        nation_kr = nation_dict[key][1]
        raw_filename = 'variation2.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        try:
            raw_df = pd.read_excel(raw_filepath, sheet_name=f'{nation}')
        except Exception as e:
            continue    
        variants = raw_df.dropna(axis=0,how='any')['Dom'].unique().tolist()
        for variant in variants:
            if variant not in variants_vocab.keys():
                variants_vocab[variant] = len(variants_vocab)
    
    # df template
    data = {
        'date': [start_date + pd.Timedelta(i, unit='D')for i in range(num_days)],
        'confirmation': [np.nan] * num_days,
        'inoculation': [np.nan] * num_days,
        'dead': [np.nan] * num_days,
        'temperature': [np.nan] * num_days,
        'humidity': [np.nan] * num_days,
        'precipitation': [np.nan] * num_days,
        'stringency': [np.nan] * num_days,
        'is_holiday': [0] * num_days,
    }

    for variant in variants_vocab.keys():
        data[variant] = [0] * num_days

    # format for meta_df
    meta_data = {
        'population': [0],
        'min_confirmation': [0],
        'max_confirmation': [0],
        'min_inoculation': [0],
        'max_inoculation': [0],
        'min_dead': [0],
        'max_dead': [0]
    }

    df_format = pd.DataFrame(data)
    meta_df_format = pd.DataFrame(meta_data)

    # preprocess holidays (weekends included)
    date_range = pd.date_range(start_date, pred_end_date)
    weekend_dates = date_range[(date_range.weekday == 5) | (date_range.weekday == 6)]
    for weekend_date in weekend_dates:
        df_format.loc[df_format['date'] == weekend_date, 'is_holiday'] = 1

    # preprocess each nation
    for key in nation_dict.keys():
        filename = f'{prp_root}/{key}.csv'
        meta_filename = f'{prp_root}/meta_{key}.csv'
        nation = nation_dict[key][0]
        nation_kr = nation_dict[key][1]
        print(nation, nation_kr)
        nation_df = copy.deepcopy(df_format)
        meta_df = copy.deepcopy(meta_df_format)

        ## 0. population
        raw_filename = 'population.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_excel(raw_filepath)
        population = raw_df[raw_df['국가'] == nation]['인구수'].item()
        
        ## 1. holiday
        raw_filename = 'dayoff.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_excel(raw_filepath)
        raw_df['날짜'] = pd.to_datetime(raw_df['날짜'])
        nation_holiday = raw_df[raw_df['국가'] == nation]
        for date in nation_holiday['날짜'].unique()[::-1]:
            if date < pred_end_date:
                # date = date.astype('datetime64[D]')
                nation_df.loc[nation_df['date'] == date, 'is_holiday'] = 1
                
        ## 2. confirmation
        # When retrieving confirmed case data, if there are 4 or more consecutive days filled with 0s, distribute the average of the values prior to the 0s.
        # However, if there are missing values (-999), leave them as is.
        raw_filename = 'confirmation.csv'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_csv(raw_filepath)
        
        sum_val = 0
        zero_dates = []

        for i, date in enumerate(raw_df['날짜']):
            confirm_num = raw_df.loc[raw_df['날짜'] == date, nation_kr].item()
            if confirm_num > 0:
                if len(zero_dates) > 4:
                    for zero_date in zero_dates:
                        nation_df.loc[nation_df['date'] == zero_date, 'confirmation'] = sum_val / len(zero_dates)
                zero_dates = [np.datetime64(date)]
                nation_df.loc[nation_df['date'] == date, 'confirmation'] = confirm_num
                sum_val = confirm_num
            if confirm_num == 0 and i > len(raw_df) * 0.5:
                zero_dates.append(np.datetime64(date))

        ## 3. dead
        # When retrieving death data, if there are 4 or more consecutive days filled with 0s, distribute the average of the values prior to the 0s.
        # However, if there are missing values (-999), leave them as is.
        raw_filename = 'dead.csv'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_csv(raw_filepath)

        sum_val = 0
        zero_dates = []

        for date in raw_df['날짜']:
            dead_num = raw_df.loc[raw_df['날짜'] == date, nation_kr].item()
            if dead_num > 0:
                if len(zero_dates) > 4:
                    for zero_date in zero_dates:
                        nation_df.loc[nation_df['date'] == zero_date, 'dead'] = sum_val / len(zero_dates)
                zero_dates = [np.datetime64(date)]
                nation_df.loc[nation_df['date'] == date, 'dead'] = dead_num
                sum_val = dead_num
            
            if dead_num == 0 and i > len(raw_df) * 0.5:
                zero_dates.append(np.datetime64(date))


        ## 4. inoculation
        # If the number of fully vaccinated individuals is greater than 0, retrieve that value, but if there are missing values (-999), leave them as NAs.
        raw_filename = 'inoculation.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_excel(raw_filepath, sheet_name=f'코로나19_접종현황_{nation_kr}')
        
        for date in raw_df['날짜']:
            inoculation = raw_df.loc[raw_df['날짜'] == date, '접종완료자'].item()
            try:
                inoculation = int(inoculation)
            except Exception as e:
                continue
            if inoculation >  0:
                nation_df.loc[nation_df['date'] == date, 'inoculation'] = inoculation
                
        ## 5. weather
        # if there are missing values (-999), leave them as NA.
        raw_filename = 'weather.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_excel(raw_filepath, sheet_name=f'코로나19_날씨_{nation_kr}')
        for date in raw_df['날짜']:
            temperature = raw_df.loc[raw_df['날짜'] == date, '평균기온'].item()
            humidity = raw_df.loc[raw_df['날짜'] == date, '평균습도'].item()
            precipitation = raw_df.loc[raw_df['날짜'] == date, '강수량'].item()
            
            if humidity > 0:
                nation_df.loc[nation_df['date'] == date, 'temperature'] = temperature
                nation_df.loc[nation_df['date'] == date, 'humidity'] = humidity
                nation_df.loc[nation_df['date'] == date, 'precipitation'] = precipitation

        ## 6. stringency index
        # Leave (-999) as NA
        # In the case of the UK and the US, if multiple strictness indices exist for the same date across different states, use the average of those values.
        raw_filename = 'stringency_index.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        raw_df = pd.read_excel(raw_filepath, sheet_name=f'코로나19_엄격성지수_{nation_kr}')
        for date in raw_df['날짜']:
            stringency = raw_df.loc[raw_df['날짜'] == date, '엄격성지수'].mean()
            if stringency > 0:
                nation_df.loc[nation_df['date'] == date, 'stringency'] = stringency

        nation_df = nation_df.interpolate(method='linear', limit_direction='both')
        
        ## 7. variants
        raw_filename = 'variation2.xlsx'
        raw_filepath = os.path.join(raw_root, raw_filename)
        if key in ['japan', 'taiwan']:
            raw_df = pd.read_excel(raw_filepath, sheet_name='South Korea')
        
        elif key == 'france':
            raw_df = pd.read_excel(raw_filepath, sheet_name='United Kingdom')

        elif key == 'italy':
            raw_df = pd.read_excel(raw_filepath, sheet_name='Germany')

        else:
            raw_df = pd.read_excel(raw_filepath, sheet_name=f'{nation}')

        dates_list = []
        dom_list = []
        mxshare_list = []

        # dominant variant information exist every 1 week, imputate 6 days
        raw_df['Dom']=raw_df['Dom'].shift().fillna(raw_df['Dom'].bfill())

        for i, row in raw_df.iterrows():
            year_week, mxshare, _, dom = row
            year = year_week.split('-')[0]
            week = year_week.split('-')[1]
            start_date = pd.to_datetime(f'{year}-W{week}-1', format='%Y-W%W-%w')
            
            date_range = pd.date_range(start=start_date, periods=7)
            dates = []
            for date in date_range:
                dates.append(date)
            dates_list.append(dates)
            dom_list.append(dom)
            mxshare_list.append(mxshare)

        df_dates = nation_df['date'].tolist()
        variants_list = variants_vocab.keys()
        for dates, dom, mxshare in zip(dates_list, dom_list, mxshare_list):
            for date in dates:
                if date in df_dates:
                    for variant in variants_list:
                        if variant == dom:
                            nation_df.loc[nation_df['date'] == date, dom] = mxshare
                        else:
                            nation_df.loc[nation_df['date'] == date, variant] = (1 - mxshare) / (len(variants_list) - 1)
                            
        # min max save
        meta_df['population'] = population
        meta_df['min_confirmation'] = int(min(nation_df['confirmation']))
        meta_df['max_confirmation'] = int(max(nation_df['confirmation']))
        meta_df['min_dead'] = int(min(nation_df['dead']))
        meta_df['max_dead'] = int(max(nation_df['dead']))
        meta_df['min_inoculation'] = min(nation_df['inoculation'])
        meta_df['max_inoculation'] = max(nation_df['inoculation'])

        # population normalizing
        nation_df['dead'] = nation_df['dead'] / population
        nation_df['confirmation'] = nation_df['confirmation'] / population
        nation_df['inoculation'] = nation_df['inoculation'] / population
        nation_df.to_csv(filename, index=False)
        meta_df.to_csv(meta_filename, index=False)

