-- Find out how many results are there within each experiment with a certain flag
SELECT
COUNT(*), exps.experiment_name
FROM experimentresults AS er
RIGHT JOIN experiments AS exps
On exps.id = er.exp_id
WHERE er.flag='success'
GROUP BY exps.id
