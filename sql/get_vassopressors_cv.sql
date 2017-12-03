select
  mimiciii.inputevents_cv.icustay_id, mimiciii.inputevents_cv.charttime, itemid, stopped, rate, amount
from mimiciii.inputevents_cv 
INNER JOIN public.rl_cohort
  ON public.rl_cohort.icustay_id = mimiciii.inputevents_cv.icustay_id
where itemid in
(
  30047,30120 -- norepinephrine
  ,30044,30119,30309 -- epinephrine
  ,30127,30128 -- phenylephrine
  ,30051 -- vasopressin
  ,30043,30307,30125 -- dopamine
)
