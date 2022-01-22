"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[53],{1109:function(e){e.exports=JSON.parse('{"pluginId":"default","version":"current","label":"Next","banner":null,"badge":false,"className":"docs-version-current","isLast":true,"docsSidebars":{"tutorialSidebar":[{"type":"link","label":"Quick start","href":"/fseval/docs/quick-start","docId":"quick-start"},{"type":"link","label":"The pipeline","href":"/fseval/docs/the-pipeline","docId":"the-pipeline"},{"type":"category","label":"fseval.config","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"PipelineConfig","href":"/fseval/docs/configuring-experiments/pipeline-config","docId":"configuring-experiments/pipeline-config"},{"type":"link","label":"DatasetConfig","href":"/fseval/docs/configuring-experiments/dataset-config","docId":"configuring-experiments/dataset-config"},{"type":"link","label":"EstimatorConfig","href":"/fseval/docs/configuring-experiments/estimator-config","docId":"configuring-experiments/estimator-config"},{"type":"link","label":"CrossValidatorConfig","href":"/fseval/docs/configuring-experiments/cross-validator-config","docId":"configuring-experiments/cross-validator-config"},{"type":"link","label":"ResampleConfig","href":"/fseval/docs/configuring-experiments/resample-config","docId":"configuring-experiments/resample-config"},{"type":"category","label":"Callbacks","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"To SQL","href":"/fseval/docs/configuring-experiments/callbacks/to_sql","docId":"configuring-experiments/callbacks/to_sql"},{"type":"link","label":"Weights and Biases","href":"/fseval/docs/configuring-experiments/callbacks/wandb","docId":"configuring-experiments/callbacks/wandb"},{"type":"link","label":"Writing your own Callback","href":"/fseval/docs/configuring-experiments/callbacks/custom","docId":"configuring-experiments/callbacks/custom"}]},{"type":"category","label":"Metrics","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Feature importance","href":"/fseval/docs/configuring-experiments/metrics/feature_importance","docId":"configuring-experiments/metrics/feature_importance"}]}]},{"type":"category","label":"Running experiments","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Running your first experiment","href":"/fseval/docs/running-experiments/running-first-experiment","docId":"running-experiments/running-first-experiment"},{"type":"link","label":"Running on a SLURM cluster","href":"/fseval/docs/running-experiments/running-on-slurm","docId":"running-experiments/running-on-slurm"}]},{"type":"category","label":"Evaluating experiments","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Analyze algorithm stability","href":"/fseval/docs/evaluating-experiments/algorithm-stability","docId":"evaluating-experiments/algorithm-stability"}],"href":"/fseval/docs/evaluating-experiments/"},{"type":"category","label":"fseval.types","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"CacheUsage","href":"/fseval/docs/types/cache-usage","docId":"types/cache-usage"}]}]},"docs":{"configuring-experiments/callbacks/custom":{"id":"configuring-experiments/callbacks/custom","title":"Writing your own Callback","description":"","sidebar":"tutorialSidebar"},"configuring-experiments/callbacks/to_sql":{"id":"configuring-experiments/callbacks/to_sql","title":"To SQL","description":"","sidebar":"tutorialSidebar"},"configuring-experiments/callbacks/wandb":{"id":"configuring-experiments/callbacks/wandb","title":"Weights and Biases","description":"","sidebar":"tutorialSidebar"},"configuring-experiments/cross-validator-config":{"id":"configuring-experiments/cross-validator-config","title":"CrossValidatorConfig","description":"Cross Validation is used to improve the reliability of the ranking and validation results. A CV method can be defined like so:","sidebar":"tutorialSidebar"},"configuring-experiments/dataset-config":{"id":"configuring-experiments/dataset-config","title":"DatasetConfig","description":"Datasets can be loaded with well-defined dataset configs. The dataset config looks as follows:","sidebar":"tutorialSidebar"},"configuring-experiments/estimator-config":{"id":"configuring-experiments/estimator-config","title":"EstimatorConfig","description":"Both feature rankers and validators are defined using the EstimatorConfig. The config for both is like below:","sidebar":"tutorialSidebar"},"configuring-experiments/metrics/feature_importance":{"id":"configuring-experiments/metrics/feature_importance","title":"Feature importance","description":"","sidebar":"tutorialSidebar"},"configuring-experiments/pipeline-config":{"id":"configuring-experiments/pipeline-config","title":"PipelineConfig","description":"All the pipeline needs to run is a well-defined configuration. The requirement is that whatever is passed into run_pipeline is a PipelineConfig object.","sidebar":"tutorialSidebar"},"configuring-experiments/resample-config":{"id":"configuring-experiments/resample-config","title":"ResampleConfig","description":"","sidebar":"tutorialSidebar"},"evaluating-experiments/algorithm-stability":{"id":"evaluating-experiments/algorithm-stability","title":"Analyze algorithm stability","description":"","sidebar":"tutorialSidebar"},"evaluating-experiments/evaluating-experiments":{"id":"evaluating-experiments/evaluating-experiments","title":"Examples of evaluating experiments","description":"evaluate!","sidebar":"tutorialSidebar"},"quick-start":{"id":"quick-start","title":"Quick start","description":"fseval helps you benchmark Feature Selection and Feature Ranking algorithms. Any algorithm that ranks features in importance.","sidebar":"tutorialSidebar"},"running-experiments/running-first-experiment":{"id":"running-experiments/running-first-experiment","title":"Running your first experiment","description":"lets go!","sidebar":"tutorialSidebar"},"running-experiments/running-on-slurm":{"id":"running-experiments/running-on-slurm","title":"Running on a SLURM cluster","description":"","sidebar":"tutorialSidebar"},"the-pipeline":{"id":"the-pipeline","title":"The pipeline","description":"fseval executes a predefined number of steps.","sidebar":"tutorialSidebar"},"types/cache-usage":{"id":"types/cache-usage","title":"CacheUsage","description":"","sidebar":"tutorialSidebar"}}}')}}]);