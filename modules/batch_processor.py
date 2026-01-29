import pandas as pd
import io

class BatchProcessor:
    def __init__(self, model, scaler, imputer=None):
        self.model = model
        self.scaler = scaler
        self.imputer = imputer
        # ATBAD 模型的标准输入顺序 (必须与 features.txt 一致)
        self.expected_cols = [
            'age', 'HR', 'BUN', 'coronary heart disease', 
            'HGB', 'hospitalization', 'renal dysfunction'
        ]

    def process_data(self, df):
        # 1. 检查列名
        missing = [c for c in self.expected_cols if c not in df.columns]
        if missing:
            return None, f"Missing columns: {missing}"
        
        # 2. 提取特征矩阵
        X_raw = df[self.expected_cols]
        
        try:
            # 3. 预处理 (如果有 imputer 就插补，没有就直接 scale)
            if self.imputer:
                X_imp = self.imputer.transform(X_raw)
                X_scl = self.scaler.transform(X_imp)
            else:
                # 兼容旧逻辑：直接 scale
                X_scl = self.scaler.transform(X_raw)
            
            # 4. 预测
            # 注意：SVM 的 predict_proba
            probs = self.model.predict_proba(X_scl)[:, 1]
            
            # 5. 拼接结果
            res_df = df.copy()
            res_df['Mortality_Risk_Prob'] = probs
            # 假设阈值 0.5 (如果不知，暂时设0.5，后面 UI 里可以调)
            res_df['Risk_Level'] = res_df['Mortality_Risk_Prob'].apply(lambda x: "High Risk" if x >= 0.5 else "Low Risk")
            
            return res_df, None
            
        except Exception as e:
            return None, f"Computation Error: {str(e)}"

    def convert_to_csv(self, df):
        return df.to_csv(index=False).encode('utf-8')

    def convert_to_excel(self, df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output
