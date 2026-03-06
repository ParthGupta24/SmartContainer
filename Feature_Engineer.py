import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import sklearn as skl

class FeatureEngineer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)

    def engineer_features(self):
        # Fix date format
        self.df['Declaration_Date'] = pd.to_datetime(self.df['Declaration_Date (YYYY-MM-DD)'], errors='coerce')

        # Time features
        self.df['hour'] = pd.to_datetime(self.df['Declaration_Time'], format='%H:%M:%S').dt.hour
        self.df['day_of_week'] = self.df['Declaration_Date'].dt.dayofweek

        # Route feature
        self.df['route'] = self.df['Origin_Country'] + "_" + self.df['Destination_Country']
        self.df['weight_diff'] = abs(self.df['Declared_Weight'] - self.df['Measured_Weight'])
        self.df['weight_ratio'] = self.df['Measured_Weight'] / (self.df['Declared_Weight'] + 1e-6)
        self.df['value_per_kg'] = self.df['Declared_Value'] / (self.df['Declared_Weight'] + 1e-6)
        importer_stats = self.df.groupby('Importer_ID').agg(
            importer_avg_value=('Declared_Value','mean'),
            importer_avg_weight=('Declared_Weight','mean'),
            importer_shipments=('Container_ID','count')
        ).reset_index()

        self.df = self.df.merge(importer_stats, on='Importer_ID', how='left')

        self.df['importer_value_dev'] = self.df['Declared_Value'] - self.df['importer_avg_value']
        exporter_stats = self.df.groupby('Exporter_ID').agg(
            exporter_avg_value=('Declared_Value','mean'),
            exporter_shipments=('Container_ID','count')
        ).reset_index()

        self.df = self.df.merge(exporter_stats, on='Exporter_ID', how='left')

        self.df['exporter_value_dev'] = self.df['Declared_Value'] - self.df['exporter_avg_value']
        route_freq = self.df['route'].value_counts().to_dict()
        self.df['route_frequency'] = self.df['route'].map(route_freq)
        hs_stats = self.df.groupby('HS_Code').agg(
            hs_avg_value_per_kg=('value_per_kg','mean')
        ).reset_index()

        self.df = self.df.merge(hs_stats,on='HS_Code',how='left')

        self.df['hs_value_deviation'] = self.df['value_per_kg'] - self.df['hs_avg_value_per_kg']
        self.df['dwell_zscore'] = (self.df['Dwell_Time_Hours'] - self.df['Dwell_Time_Hours'].mean()) / self.df['Dwell_Time_Hours'].std()
        categorical_cols = [
            'Trade_Regime (Import / Export / Transit)',
            'Origin_Country',
            'Destination_Country',
            'Destination_Port',
            'Shipping_Line',
            'route'
        ]

        le_dict = {}

        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            le_dict[col] = le

        self.df['Clear/Not Clear'] = self.df['Clearance_Status'].apply(lambda x : 1 if x != 'Clear' else 0)
        self.df['Clearance_Status'] = self.df.loc[:, 'Clearance_Status'].map({'Clear': 0, 'Low Risk': 1, 'Critical': 2})
        features = [
        'Declared_Value',
        'Declared_Weight',
        'Measured_Weight',
        'Dwell_Time_Hours',

        'weight_diff',
        'weight_ratio',
        'value_per_kg',

        'importer_avg_value',
        'importer_shipments',
        'importer_value_dev',

        'exporter_avg_value',
        'exporter_shipments',
        'exporter_value_dev',

        'route_frequency',
        'hs_value_deviation',
        'dwell_zscore',

        'hour',
        'day_of_week',

        'Trade_Regime (Import / Export / Transit)',
        'Origin_Country',
        'Destination_Country',
        'Destination_Port',
        'Shipping_Line',
        'route'
        ]

        X = self.df[features]
        y1 = self.df['Clear/Not Clear']
        y2 = self.df['Clearance_Status']
        return X, y1, y2
