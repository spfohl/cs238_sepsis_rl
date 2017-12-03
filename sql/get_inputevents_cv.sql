SELECT input.subject_id, input.hadm_id, 
input.icustay_id, input.charttime, 
input.itemid, input.amount, 
input.amountuom, input.rate, 
input.rateuom, input.storetime, input.orderid
FROM mimiciii.inputevents_cv input 
INNER JOIN public.rl_cohort on input.subject_id = public.rl_cohort.subject_id 
ORDER BY subject_id ASC