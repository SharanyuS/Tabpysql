from Abs_LR_module import absenteeism_model,CustomScaler

model = absenteeism_model('Abs_model','Abs_scaler')

model.load_and_clean_data('Absenteeism_new_data.csv')

model.predicted_outputs()







