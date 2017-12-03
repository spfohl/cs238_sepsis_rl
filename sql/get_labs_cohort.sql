SELECT rl_cohort.subject_id, rl_cohort.hadm_id, rl_cohort.icustay_id,
	 window_start, window_end, intime, outtime, charttime, lab_id, valuenum
FROM public.rl_cohort
INNER JOIN mimiciii.labs_all_rl l
	ON l.subject_id = rl_cohort.subject_id
WHERE l.charttime >= rl_cohort.window_start AND 
		l.charttime <= rl_cohort.window_end;