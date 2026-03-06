## Description:
<br><br>
SmartContainer library contains classes and methods for feature engineering and ML model operations on Cargo Container Datasets in format as provided in HackAMineD 2026, Track - InTech (organized by Nirma University). The Modules available in this library are as follows:
<br><hr><br>
### Feature_Engineer
<br><br>
Feature_Engineer handles Feature Engineeering operations.
class FeatureEngineer(file_path): base class for the module; holds the dataset for purpose of data manipulation

    self.file_path: contains the file path of the dataset as entered by the user (in string format)
    self.df: pandas dataframe of the input dataset. Dataset must be in .CSV format
    self.engineer_features(): Performs feature engineering operations. Affects self.df - (self.df now has both - original features and engineered features. all features are LabelEncoded using sklearn)
        inputs - None
        outputs -  X : input features for ML model predictions
                  y1 : target features in categories (Clear / Not Clear)
                  y2 : target features in categories (No Risk / Low Risk / High Risk)
<br>
### Models

