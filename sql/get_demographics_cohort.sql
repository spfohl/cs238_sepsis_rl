SELECT r.subject_id, r.hadm_id, r.icustay_id, age, is_male, 
		race_white, race_black, race_hispanic, race_other, 
        height, weight, vent, sofa, lods, sirs, 
        qsofa, qsofa_sysbp_score, qsofa_gcs_score, qsofa_resprate_score, 
        elixhauser_hospital, blood_culture_positive
FROM public.rl_cohort r
INNER JOIN public.sepsis3 
	ON r.icustay_id = sepsis3.icustay_id;