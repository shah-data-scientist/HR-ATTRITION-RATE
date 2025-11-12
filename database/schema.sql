CREATE TABLE employees (
	id_employee SERIAL NOT NULL, 
	age INTEGER, 
	genre INTEGER, 
	revenu_mensuel FLOAT, 
	statut_marital VARCHAR, 
	departement VARCHAR, 
	poste VARCHAR, 
	nombre_experiences_precedentes INTEGER, 
	annee_experience_totale INTEGER, 
	annees_dans_l_entreprise FLOAT, 
	annees_dans_le_poste_actuel FLOAT, 
	nombre_participation_pee INTEGER, 
	nb_formations_suivies INTEGER, 
	nombre_employee_sous_responsabilite INTEGER, 
	distance_domicile_travail INTEGER, 
	niveau_education INTEGER, 
	domaine_etude VARCHAR, 
	ayant_enfants INTEGER, 
	frequence_deplacement VARCHAR, 
	annees_depuis_la_derniere_promotion INTEGER, 
	annes_sous_responsable_actuel INTEGER, 
	satisfaction_employee_environnement FLOAT, 
	note_evaluation_precedente FLOAT, 
	niveau_hierarchique_poste FLOAT, 
	satisfaction_employee_nature_travail FLOAT, 
	satisfaction_employee_equipe FLOAT, 
	satisfaction_employee_equilibre_pro_perso FLOAT, 
	note_evaluation_actuelle FLOAT, 
	heures_supplementaires INTEGER, 
	augmentation_salaire_precedente FLOAT, 
	augementation_salaire_precedente FLOAT, 
	nombre_heures_travailless FLOAT, 
	improvement_evaluation FLOAT, 
	total_satisfaction FLOAT, 
	work_mobility FLOAT, 
	date_ingestion TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	PRIMARY KEY (id_employee)
);

CREATE TABLE model_outputs (
	output_id SERIAL NOT NULL, 
	prediction_proba FLOAT NOT NULL, 
	risk_category VARCHAR NOT NULL, 
	prediction_label VARCHAR NOT NULL, 
	log_odds FLOAT NOT NULL, 
	prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	PRIMARY KEY (output_id)
);

CREATE TABLE model_inputs (
	input_id SERIAL NOT NULL, 
	id_employee INTEGER NOT NULL, 
	prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	features JSON NOT NULL, 
	PRIMARY KEY (input_id), 
	FOREIGN KEY(id_employee) REFERENCES employees (id_employee)
);

CREATE TABLE predictions_traceability (
	trace_id SERIAL NOT NULL, 
	input_id INTEGER NOT NULL, 
	output_id INTEGER NOT NULL, 
	model_version VARCHAR, 
	prediction_source VARCHAR, 
	request_metadata JSON, 
	created_at TIMESTAMP WITH TIME ZONE DEFAULT now(), 
	PRIMARY KEY (trace_id), 
	FOREIGN KEY(input_id) REFERENCES model_inputs (input_id), 
	FOREIGN KEY(output_id) REFERENCES model_outputs (output_id)
);

