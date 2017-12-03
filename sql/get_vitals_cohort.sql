SELECT rl_cohort.subject_id, rl_cohort.hadm_id, rl_cohort.icustay_id,
	 window_start, window_end, intime, outtime, charttime, vital_id, valuenum
FROM public.rl_cohort
INNER JOIN mimiciii.vitals_all_rl v
	ON v.subject_id = rl_cohort.subject_id
WHERE v.charttime >= rl_cohort.window_start AND 
		v.charttime <= rl_cohort.window_end;