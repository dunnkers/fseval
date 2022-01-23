"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[53],{1109:function(e){e.exports=JSON.parse('{"pluginId":"default","version":"current","label":"Next","banner":null,"badge":false,"className":"docs-version-current","isLast":true,"docsSidebars":{"tutorialSidebar":[{"type":"link","label":"Quick start","href":"/fseval/docs/quick-start","docId":"quick-start"},{"type":"link","label":"The pipeline","href":"/fseval/docs/the-pipeline","docId":"the-pipeline"},{"type":"category","label":"fseval.config","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"fseval.config.PipelineConfig","href":"/fseval/docs/config/PipelineConfig","docId":"config/PipelineConfig"},{"type":"category","label":"fseval.config.callbacks","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"ToSQLCallback","href":"/fseval/docs/config/callbacks/to_sql","docId":"config/callbacks/to_sql"},{"type":"link","label":"WandbCallback","href":"/fseval/docs/config/callbacks/wandb","docId":"config/callbacks/wandb"},{"type":"link","label":"\u2699\ufe0f Custom Callbacks","href":"/fseval/docs/config/callbacks/custom","docId":"config/callbacks/custom"}]},{"type":"link","label":"fseval.config.DatasetConfig","href":"/fseval/docs/config/DatasetConfig","docId":"config/DatasetConfig"},{"type":"category","label":"fseval.config.adapters","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"OpenMLDataset","href":"/fseval/docs/config/adapters/OpenMLDataset","docId":"config/adapters/OpenMLDataset"},{"type":"link","label":"WandbDataset","href":"/fseval/docs/config/adapters/WandbDataset","docId":"config/adapters/WandbDataset"},{"type":"link","label":"\u2699\ufe0f Custom Adapters","href":"/fseval/docs/config/adapters/Custom","docId":"config/adapters/Custom"}]},{"type":"link","label":"fseval.config.CrossValidatorConfig","href":"/fseval/docs/config/CrossValidatorConfig","docId":"config/CrossValidatorConfig"},{"type":"link","label":"fseval.config.ResampleConfig","href":"/fseval/docs/config/ResampleConfig","docId":"config/ResampleConfig"},{"type":"link","label":"fseval.config.EstimatorConfig","href":"/fseval/docs/config/EstimatorConfig","docId":"config/EstimatorConfig"},{"type":"link","label":"fseval.config.StorageConfig","href":"/fseval/docs/config/StorageConfig","docId":"config/StorageConfig"},{"type":"category","label":"fseval.config.storage","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"LocalStorageConfig","href":"/fseval/docs/config/storage/local","docId":"config/storage/local"}]},{"type":"category","label":"fseval.config.metrics","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"FeatureImportances","href":"/fseval/docs/config/metrics/feature_importance","docId":"config/metrics/feature_importance"},{"type":"link","label":"RankingScores","href":"/fseval/docs/config/metrics/ranking_scores","docId":"config/metrics/ranking_scores"},{"type":"link","label":"ValidationScores","href":"/fseval/docs/config/metrics/validation_scores","docId":"config/metrics/validation_scores"}]}]},{"type":"category","label":"fseval.types","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"fseval.types.CacheUsage","href":"/fseval/docs/types/CacheUsage","docId":"types/CacheUsage"},{"type":"link","label":"fseval.types.Task","href":"/fseval/docs/types/Task","docId":"types/Task"}]},{"type":"category","label":"Running experiments","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Running your first experiment","href":"/fseval/docs/running-experiments/running-first-experiment","docId":"running-experiments/running-first-experiment"},{"type":"link","label":"Running on a SLURM cluster","href":"/fseval/docs/running-experiments/running-on-slurm","docId":"running-experiments/running-on-slurm"}]},{"type":"category","label":"Evaluating experiments","collapsible":true,"collapsed":true,"items":[{"type":"link","label":"Analyze algorithm stability","href":"/fseval/docs/evaluating-experiments/algorithm-stability","docId":"evaluating-experiments/algorithm-stability"}],"href":"/fseval/docs/evaluating-experiments/"}]},"docs":{"config/adapters/Custom":{"id":"config/adapters/Custom","title":"\u2699\ufe0f Custom Adapters","description":"To load datasets from different sources, we can use different adapters. You can create an adapter by implementing this interface:","sidebar":"tutorialSidebar"},"config/adapters/OpenMLDataset":{"id":"config/adapters/OpenMLDataset","title":"OpenMLDataset","description":"Allows loading a dataset from OpenML.","sidebar":"tutorialSidebar"},"config/adapters/WandbDataset":{"id":"config/adapters/WandbDataset","title":"WandbDataset","description":"Loads a dataset from the Weights and Biases artifacts store.","sidebar":"tutorialSidebar"},"config/callbacks/custom":{"id":"config/callbacks/custom","title":"\u2699\ufe0f Custom Callbacks","description":"","sidebar":"tutorialSidebar"},"config/callbacks/to_sql":{"id":"config/callbacks/to_sql","title":"ToSQLCallback","description":"","sidebar":"tutorialSidebar"},"config/callbacks/wandb":{"id":"config/callbacks/wandb","title":"WandbCallback","description":"","sidebar":"tutorialSidebar"},"config/CrossValidatorConfig":{"id":"config/CrossValidatorConfig","title":"fseval.config.CrossValidatorConfig","description":"Cross Validation is used to improve the reliability of the ranking and validation results. A CV method can be defined like so:","sidebar":"tutorialSidebar"},"config/DatasetConfig":{"id":"config/DatasetConfig","title":"fseval.config.DatasetConfig","description":"Configures a dataset, to be used in the pipeline. Can be loaded from various sources","sidebar":"tutorialSidebar"},"config/EstimatorConfig":{"id":"config/EstimatorConfig","title":"fseval.config.EstimatorConfig","description":"Configures an estimator: a Feature Ranker, Feature Selector or a validation","sidebar":"tutorialSidebar"},"config/metrics/feature_importance":{"id":"config/metrics/feature_importance","title":"FeatureImportances","description":"","sidebar":"tutorialSidebar"},"config/metrics/ranking_scores":{"id":"config/metrics/ranking_scores","title":"RankingScores","description":"","sidebar":"tutorialSidebar"},"config/metrics/validation_scores":{"id":"config/metrics/validation_scores","title":"ValidationScores","description":"","sidebar":"tutorialSidebar"},"config/PipelineConfig":{"id":"config/PipelineConfig","title":"fseval.config.PipelineConfig","description":"\x3c!-- All the pipeline needs to run is a well-defined configuration. The requirement is that whatever is passed into run_pipeline is a PipelineConfig object.","sidebar":"tutorialSidebar"},"config/ResampleConfig":{"id":"config/ResampleConfig","title":"fseval.config.ResampleConfig","description":"Resampling can be used to take random samples from the dataset, with- or without replacement. Resampling is performed after the CV split.","sidebar":"tutorialSidebar"},"config/storage/local":{"id":"config/storage/local","title":"LocalStorageConfig","description":"","sidebar":"tutorialSidebar"},"config/StorageConfig":{"id":"config/StorageConfig","title":"fseval.config.StorageConfig","description":"Allows you to define a storage for loading and saving cached estimators, among other","sidebar":"tutorialSidebar"},"evaluating-experiments/algorithm-stability":{"id":"evaluating-experiments/algorithm-stability","title":"Analyze algorithm stability","description":"","sidebar":"tutorialSidebar"},"evaluating-experiments/evaluating-experiments":{"id":"evaluating-experiments/evaluating-experiments","title":"Examples of evaluating experiments","description":"evaluate!","sidebar":"tutorialSidebar"},"quick-start":{"id":"quick-start","title":"Quick start","description":"fseval helps you benchmark Feature Selection and Feature Ranking algorithms. Any algorithm that ranks features in importance.","sidebar":"tutorialSidebar"},"running-experiments/running-first-experiment":{"id":"running-experiments/running-first-experiment","title":"Running your first experiment","description":"lets go!","sidebar":"tutorialSidebar"},"running-experiments/running-on-slurm":{"id":"running-experiments/running-on-slurm","title":"Running on a SLURM cluster","description":"","sidebar":"tutorialSidebar"},"the-pipeline":{"id":"the-pipeline","title":"The pipeline","description":"fseval executes a predefined number of steps.","sidebar":"tutorialSidebar"},"types/CacheUsage":{"id":"types/CacheUsage","title":"fseval.types.CacheUsage","description":"Determines how cache usage is handled. In the case of loading caches:","sidebar":"tutorialSidebar"},"types/Task":{"id":"types/Task","title":"fseval.types.Task","description":"Learning task. In the case of datasets this indicates the dataset learning task, and in the case of estimators this indicates the supported estimator learning tasks.","sidebar":"tutorialSidebar"}}}')}}]);