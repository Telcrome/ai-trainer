SELECT DISTINCT
experimentresults.name
FROM
experimentresults
JOIN
experiments
ON
experiments.id = experimentresults.exp_id
WHERE
	(experiments.start_date < '2020-06-20'
	 AND experiments.start_date > '2020-06-19'
	 AND
	 	(exp_id = 61 OR exp_id = 60 OR exp_id = 59 OR exp_id = 62)
	 AND flag = 'success'
	 )
