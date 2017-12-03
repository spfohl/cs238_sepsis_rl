psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_cohort.sql -o ./../data/cohort.csv

psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_vassopressors_cv.sql -o ./../data/vassopressors_cv_cohort.csv
psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_vassopressors_mv.sql -o ./../data/vassopressors_mv_cohort.csv

psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_inputevents_cv.sql -o ./../data/inputevents_cv_cohort.csv
psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_inputevents_mv.sql -o ./../data/inputevents_mv_cohort.csv

psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_labs_cohort.sql -o ./../data/labs_cohort.csv
psql -d mimic -U stephenpfohl --pset="footer=off" -A -F , -f get_vitals_cohort.sql -o ./../data/vitals_cohort.csv