DROP MATERIALIZED VIEW IF EXISTS rl_cohort CASCADE;
CREATE MATERIALIZED VIEW rl_cohort as

SELECT DISTINCT subject_id, temp.hadm_id, temp.icustay_id, 
			intime, outtime, suspected_infection_time_poe,
            suspected_infection_time_poe - interval '1 day' as window_start,
            suspected_infection_time_poe + interval '2 day' as window_end,
            hospital_expire_flag
            	FROM (
    SELECT s3.hadm_id, s3.icustay_id, 
            s3.suspected_infection_time_poe,
            s3.suspected_infection_time_poe_days, sofa.sofa, 
            s3.intime, s3.outtime, s3.excluded,
            s3c.exclusion_secondarystay,
            s3c.exclusion_nonadult,
            s3c.exclusion_csurg,
            s3c.exclusion_carevue,
            s3c.exclusion_early_suspicion,
            s3c.exclusion_late_suspicion,
            s3c.exclusion_bad_data
    FROM sepsis3 as s3
    INNER JOIN  sofa 
    	ON s3.hadm_id = sofa.hadm_id
    INNER JOIN sepsis3_cohort as s3c
    	on s3c.icustay_id = s3.icustay_id
    WHERE s3.suspected_infection_time_poe_days is NOT NULL AND sofa.sofa >= 2
    	-- AND excluded = 0
        AND exclusion_nonadult = 0
    	AND exclusion_bad_data = 0
    	AND exclusion_secondarystay = 0
    )
   as temp
INNER JOIN mimiciii.admissions
ON temp.hadm_id = admissions.hadm_id;