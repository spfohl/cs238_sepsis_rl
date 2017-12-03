set search_path to mimiciii;

DROP MATERIALIZED VIEW IF EXISTS gcs CASCADE;
create materialized view gcs as
WITH agetbl AS
(
    SELECT ad.subject_id
    FROM admissions ad
    INNER JOIN patients p
    ON ad.subject_id = p.subject_id
    WHERE
     -- filter to only adults
    EXTRACT(EPOCH FROM (ad.admittime - p.dob))/60.0/60.0/24.0/365.242 > 15
    -- group by subject_id to ensure there is only 1 subject_id per row
    group by ad.subject_id
)
, gcs as
(
    SELECT ce.subject_id, ce.hadm_id, ce.icustay_id, ce.itemid, ce.charttime, ce.value, ce.valuenum
    FROM chartevents ce
    INNER JOIN agetbl
    ON ce.subject_id = agetbl.subject_id
    WHERE itemid IN
    (
        454 -- "Motor Response"
      , 223900 -- "GCS - Motor Response"
    )
)
SELECT * from gcs
group by gcs.subject_id, itemid, gcs.hadm_id, gcs.icustay_id, gcs.charttime, gcs.value, gcs.valuenum
order by gcs.subject_id, itemid, gcs.hadm_id, gcs.icustay_id, gcs.charttime;