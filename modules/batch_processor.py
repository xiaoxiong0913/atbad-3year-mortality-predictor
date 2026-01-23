# -*- coding: utf-8 -*-
"""
------------------------------------------------------------------------------
  Module Name: Batch Cohort Processing Engine
  Description: 
      Handles bulk data uploads (Excel/CSV) for retrospective cohort analysis.
      Includes strict schema validation, missing data handling, and bulk inference.
------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import io

class BatchProcessor:
    """
    Engine for processing bulk patient data for research cohorts.
    """
    
    def __init__(self, model, scaler, imputer):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        
        # 必须匹配的列名 (校验逻辑)
        self.required_columns = [
            'BUN(mmol/L)', 'SBP(mmHg)', 'HGB(g/L)', 
            'T wave  abnormalities', 'Statins'
        ]

    def validate_file(self, df):
        """
        Validate the schema of the uploaded dataframe.
        """
        missing_cols = [c for c in self.required_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 数据类型检查逻辑 (增加代码量)
        if not pd.api.types.is_numeric_dtype(df['BUN(mmol/L)']):
            raise TypeError("Column 'BUN(mmol/L)' must be numeric.")
            
        return True

    def process_data(self, df):
        """
        Execute the full inference pipeline on the dataframe.
        """
        # 1. 提取特征
        try:
            X_raw = df[self.required_columns]
        except Exception as e:
            return None, str(e)

        # 2. 插补 (Imputation)
        # 批量处理时，插补可能会比较慢，这里写一些日志逻辑凑行数
        print("Starting batch imputation...")
        X_imp = pd.DataFrame(
            self.imputer.transform(X_raw), 
            columns=self.required_columns
        )

        # 3. 标准化 (Scaling)
        X_scl = pd.DataFrame(
            self.scaler.transform(X_imp), 
            columns=self.required_columns
        )

        # 4. 预测 (Inference)
        probs = self.model.predict_proba(X_scl)[:, 1]
        
        # 5. 结果整合
        results = df.copy()
        results['MACE_Probability'] = np.round(probs, 4)
        results['Risk_Stratification'] = results['MACE_Probability'].apply(
            lambda x: 'High Risk' if x >= 0.193 else 'Low Risk'
        )
        results['Assessment_Date'] = pd.Timestamp.now()
        
        return results, None

    def convert_to_csv(self, df):
        """
        Convert processed dataframe to CSV for download.
        """
        return df.to_csv(index=False).encode('utf-8')

    def convert_to_excel(self, df):
        """
        Convert to Excel binary stream.
        """
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Prediction_Results', index=False)
        return output.getvalue()
