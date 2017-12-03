  select
    mimiciii.inputevents_mv.icustay_id, linkorderid, starttime, endtime, itemid, rate, amount
  from mimiciii.inputevents_mv
  -- Subselect the vasopressor ITEMIDs
  INNER JOIN public.rl_cohort
    ON public.rl_cohort.icustay_id = mimiciii.inputevents_mv.icustay_id
  where itemid in
  (
  221906 -- norepinephrine
  ,221289 -- epinephrine
  ,221749 -- phenylephrine
  ,222315 -- vasopressin
  ,221662 -- dopamine
  )
  and statusdescription != 'Rewritten' -- only valid orders