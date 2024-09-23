for start_date in '2020-02-27'
do
/mlainas/teang1995/anaconda3/envs/mogp/bin/python baselines.py \
--seed 1000 \
--model_type 'ARIMA' \
--start_date ${start_date} \
--obs_end_date '2023-02-07' \
--pred_end_date '2023-12-31'
done