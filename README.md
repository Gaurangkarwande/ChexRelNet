# ChexRelNet

Despite the progress in utilizing deep learning to automate
chest radiograph interpretation and disease diagnosis tasks, change be-
tween sequential Chest X-rays (CXRs) has received limited attraction.
Monitoring the progression of pathologies that are visualized through
chest imaging poses several challenges in anatomical motion estimation
and image registration, i.e., spatially aligning the two images and model-
ing temporal dynamics in change detection. In this work, we propose a
novel neural model that can track longitudinal pathology change relations
between two CXRs. The proposed model incorporates both local and
global visual features, utilizes inter-image and intra-image anatomical in-
formation and learns dependencies between anatomical region attributes,
to accurately predict disease change for a pair of CXRs. Experimental
results on the Chest ImaGenome dataset showcase increased downstream
and zero-shot performance when compared to baselines