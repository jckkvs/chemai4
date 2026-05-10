# ChemAI ML Studio: Feature Matrix (星取表)

| ID | Module/File | Feature/Function | Type | Test ID | Status |
|----|-------------|------------------|------|---------|--------|
| F-001 | `preset_manager.py` | `save_preset`(name, state) | MUST | T-001 |  |
| F-002 | `preset_manager.py` | `load_preset`(name, state) | MUST | T-002 |  |
| F-003 | `preset_manager.py` | `list_presets`() | MUST | T-003 |  |
| F-004 | `preset_manager.py` | `delete_preset`(name) | MUST | T-004 |  |
| F-005 | `preset_manager.py` | `export_state_summary`(state) | MUST | T-005 |  |
| F-006 | `preset_manager.py` | `record_analysis`(state, result) | MUST | T-006 |  |
| F-007 | `preset_manager.py` | `list_history`() | MUST | T-007 |  |
| F-008 | `preset_manager.py` | `export_config_yaml`(state) | MUST | T-008 |  |
| F-009 | `preset_manager.py` | `import_config_yaml`(yaml_text, state) | MUST | T-009 |  |
| F-010 | `chem\base.py` | Class `DescriptorResult` | MUST | T-010 |  |
| F-011 | `chem\base.py` | Class `DescriptorMetadata` | MUST | T-011 |  |
| F-012 | `chem\base.py` | Class `BaseChemAdapter` | MUST | T-012 |  |
| F-013 | `chem\charge_config.py` | Class `MoleculeChargeConfig` | MUST | T-013 |  |
| F-014 | `chem\charge_config.py` | Class `ChargeConfigStore` | MUST | T-014 |  |
| F-015 | `chem\chemprop_adapter.py` | Class `ChempropAdapter` | MUST | T-015 |  |
| F-016 | `chem\cosmo_adapter.py` | Class `CosmoAdapter` | MUST | T-016 |  |
| F-017 | `chem\descriptastorus_adapter.py` | Class `DescriptaStorusAdapter` | MUST | T-017 |  |
| F-018 | `chem\descriptor_sets.py` | Class `DescriptorSet` | MUST | T-018 |  |
| F-019 | `chem\descriptor_sets.py` | Class `DescriptorSetManager` | MUST | T-019 |  |
| F-020 | `chem\group_contrib_adapter.py` | Class `_JobackGroup` | MUST | T-020 |  |
| F-021 | `chem\group_contrib_adapter.py` | Class `GroupContribAdapter` | MUST | T-021 |  |
| F-022 | `chem\mix_rules.py` | Class `MixRulesManager` | MUST | T-022 |  |
| F-023 | `chem\mix_rules.py` | `get_default_rule`(descriptor_name) | MUST | T-023 |  |
| F-024 | `chem\mol2vec_adapter.py` | Class `Mol2VecAdapter` | MUST | T-024 |  |
| F-025 | `chem\molai_adapter.py` | Class `MolAIAdapter` | MUST | T-025 |  |
| F-026 | `chem\molai_inverse.py` | Class `MolCandidate` | MUST | T-026 |  |
| F-027 | `chem\molai_inverse.py` | Class `DegeneracyMap` | MUST | T-027 |  |
| F-028 | `chem\molai_inverse.py` | Class `MolAIInverseAnalyzer` | MUST | T-028 |  |
| F-029 | `chem\molfeat_adapter.py` | Class `MolfeatAdapter` | MUST | T-029 |  |
| F-030 | `chem\mordred_adapter.py` | Class `MordredAdapter` | MUST | T-030 |  |
| F-031 | `chem\padel_adapter.py` | Class `PaDELAdapter` | MUST | T-031 |  |
| F-032 | `chem\protonation.py` | `apply_protonation`(smiles, config) | MUST | T-032 |  |
| F-033 | `chem\protonation.py` | `apply_protonation_batch`(smiles_list, config) | MUST | T-033 |  |
| F-034 | `chem\protonation.py` | `get_protonation_state_info`(smiles, ph) | MUST | T-034 |  |
| F-035 | `chem\psmiles_adapter.py` | Class `PSmilesAdapter` | MUST | T-035 |  |
| F-036 | `chem\rdkit_adapter.py` | Class `RDKitAdapter` | MUST | T-036 |  |
| F-037 | `chem\recommender.py` | Class `DescriptorInfo` | MUST | T-037 |  |
| F-038 | `chem\recommender.py` | Class `TargetRecommendations` | MUST | T-038 |  |
| F-039 | `chem\recommender.py` | `get_all_target_recommendations`() | MUST | T-039 |  |
| F-040 | `chem\recommender.py` | `get_target_recommendation_by_name`(name) | MUST | T-040 |  |
| F-041 | `chem\recommender.py` | `get_target_names`() | MUST | T-041 |  |
| F-042 | `chem\recommender.py` | `get_target_categories`() | MUST | T-042 |  |
| F-043 | `chem\recommender.py` | `get_targets_by_category`(category) | MUST | T-043 |  |
| F-044 | `chem\recommender.py` | `get_all_descriptor_categories`() | MUST | T-044 |  |
| F-045 | `chem\recommender.py` | `get_descriptors_by_category`(category) | MUST | T-045 |  |
| F-046 | `chem\skfp_adapter.py` | Class `SkfpAdapter` | MUST | T-046 |  |
| F-047 | `chem\smiles_transformer.py` | Class `SmilesDescriptorTransformer` | MUST | T-047 |  |
| F-048 | `chem\smiles_transformer.py` | `progressive_precalculate`(smiles_list, target_col_name) | MUST | T-048 |  |
| F-049 | `chem\smiles_transformer.py` | `precalculate_all_descriptors`(smiles_list, target_col_name, engine_flags, molai_n_components, progress_callback) | MUST | T-049 |  |
| F-050 | `chem\uma_adapter.py` | Class `UMAAdapter` | MUST | T-050 |  |
| F-051 | `chem\unipka_adapter.py` | Class `UniPkaAdapter` | MUST | T-051 |  |
| F-052 | `chem\xtb_adapter.py` | Class `XTBAdapter` | MUST | T-052 |  |
| F-053 | `chem\__init__.py` | `get_available_adapters`() | MUST | T-053 |  |
| F-054 | `chem\descriptors\base.py` | Class `ParamDef` | MUST | T-054 |  |
| F-055 | `chem\descriptors\base.py` | Class `PluginInfo` | MUST | T-055 |  |
| F-056 | `chem\descriptors\base.py` | `validate_plugin`(module, filepath) | MUST | T-056 |  |
| F-057 | `chem\descriptors\base.py` | `safe_compute`(plugin, smiles_list) | MUST | T-057 |  |
| F-058 | `chem\descriptors\__init__.py` | `discover_plugins`(force_reload) | MUST | T-058 |  |
| F-059 | `chem\descriptors\__init__.py` | `get_plugins_by_engine`(engine) | MUST | T-059 |  |
| F-060 | `chem\descriptors\__init__.py` | `get_plugins_by_category`(category) | MUST | T-060 |  |
| F-061 | `chem\descriptors\__init__.py` | `get_available_engines`() | MUST | T-061 |  |
| F-062 | `chem\descriptors\__init__.py` | `get_available_categories`() | MUST | T-062 |  |
| F-063 | `chem\descriptors\__init__.py` | `compute_all_descriptors`(smiles_list, plugin_names, progress_callback) | MUST | T-063 |  |
| F-064 | `chem\descriptors\__init__.py` | `get_custom_dir`() | MUST | T-064 |  |
| F-065 | `chem\descriptors\__init__.py` | `get_builtins_dir`() | MUST | T-065 |  |
| F-066 | `chem\descriptors\__init__.py` | `invalidate_cache`() | MUST | T-066 |  |
| F-067 | `chem\descriptors\custom\ext_test_lipophilicity.py` | `compute`(smiles_list) | MUST | T-067 |  |
| F-068 | `chem\descriptors\custom\_template_simple.py` | `compute`(smiles_list) | MUST | T-068 |  |
| F-069 | `chem\descriptors\custom\_template_with_config.py` | `compute`(smiles_list) | MUST | T-069 |  |
| F-070 | `chem\descriptors\_builtins\cosmo_sigma.py` | `compute`(smiles_list) | MUST | T-070 |  |
| F-071 | `chem\descriptors\_builtins\group_contrib.py` | `compute`(smiles_list) | MUST | T-071 |  |
| F-072 | `chem\descriptors\_builtins\mordred_selected.py` | `compute`(smiles_list) | MUST | T-072 |  |
| F-073 | `chem\descriptors\_builtins\rdkit_electronic.py` | `compute`(smiles_list) | MUST | T-073 |  |
| F-074 | `chem\descriptors\_builtins\rdkit_fingerprints.py` | `compute`(smiles_list) | MUST | T-074 |  |
| F-075 | `chem\descriptors\_builtins\rdkit_fragments.py` | `compute`(smiles_list) | MUST | T-075 |  |
| F-076 | `chem\descriptors\_builtins\rdkit_graph_indices.py` | `compute`(smiles_list) | MUST | T-076 |  |
| F-077 | `chem\descriptors\_builtins\rdkit_physicochemical.py` | `compute`(smiles_list) | MUST | T-077 |  |
| F-078 | `chem\descriptors\_builtins\rdkit_surface_area.py` | `compute`(smiles_list) | MUST | T-078 |  |
| F-079 | `chem\descriptors\_builtins\rdkit_topology.py` | `compute`(smiles_list) | MUST | T-079 |  |
| F-080 | `chem\descriptors\_builtins\xtb_electronic.py` | `compute`(smiles_list) | MUST | T-080 |  |
| F-081 | `data\benchmark.py` | Class `ModelScore` | MUST | T-081 |  |
| F-082 | `data\benchmark.py` | Class `BenchmarkResult` | MUST | T-082 |  |
| F-083 | `data\benchmark.py` | `evaluate_regression`(y_true, y_pred, model_key, train_time, cv_mean, cv_std) | MUST | T-083 |  |
| F-084 | `data\benchmark.py` | `evaluate_classification`(y_true, y_pred, y_prob, model_key, train_time, cv_mean, cv_std) | MUST | T-084 |  |
| F-085 | `data\benchmark.py` | `compute_learning_curve`(estimator, X, y, scoring, cv, n_points, n_jobs) | MUST | T-085 |  |
| F-086 | `data\benchmark.py` | `benchmark_models`(models, X_train, y_train, X_test, y_test, task, scoring) | MUST | T-086 |  |
| F-087 | `data\benchmark_datasets.py` | `list_benchmark_datasets`() | MUST | T-087 |  |
| F-088 | `data\benchmark_datasets.py` | `load_benchmark`(name) | MUST | T-088 |  |
| F-089 | `data\data_cleaner.py` | Class `CleaningAction` | MUST | T-089 |  |
| F-090 | `data\data_cleaner.py` | `drop_columns`(df, columns) | MUST | T-090 |  |
| F-091 | `data\data_cleaner.py` | `drop_rows_with_missing`(df, threshold, subset) | MUST | T-091 |  |
| F-092 | `data\data_cleaner.py` | `remove_constant_columns`(df) | MUST | T-092 |  |
| F-093 | `data\data_cleaner.py` | `clip_outliers`(df, iqr_multiplier, columns) | MUST | T-093 |  |
| F-094 | `data\data_cleaner.py` | `remove_duplicates`(df, subset, keep) | MUST | T-094 |  |
| F-095 | `data\data_cleaner.py` | `preview_missing_impact`(df, threshold, subset) | MUST | T-095 |  |
| F-096 | `data\data_cleaner.py` | `preview_outlier_impact`(df, iqr_multiplier, columns) | MUST | T-096 |  |
| F-097 | `data\data_cleaner.py` | `get_cleaning_summary`(df) | MUST | T-097 |  |
| F-098 | `data\dim_reduction.py` | Class `DimReductionConfig` | MUST | T-098 |  |
| F-099 | `data\dim_reduction.py` | Class `DimReducer` | MUST | T-099 |  |
| F-100 | `data\dim_reduction.py` | `run_pca`(df, n_components, scale, target_col) | MUST | T-100 |  |
| F-101 | `data\dim_reduction.py` | `run_tsne`(df, n_components, perplexity, scale, target_col, random_state) | MUST | T-101 |  |
| F-102 | `data\dim_reduction.py` | `run_umap`(df, n_components, n_neighbors, min_dist, scale, target_col, random_state) | MUST | T-102 |  |
| F-103 | `data\eda.py` | Class `ColumnStats` | MUST | T-103 |  |
| F-104 | `data\eda.py` | Class `OutlierResult` | MUST | T-104 |  |
| F-105 | `data\eda.py` | `compute_column_stats`(df) | MUST | T-105 |  |
| F-106 | `data\eda.py` | `summarize_dataframe`(df) | MUST | T-106 |  |
| F-107 | `data\eda.py` | `compute_correlation`(df, method, target_col) | MUST | T-107 |  |
| F-108 | `data\eda.py` | `detect_outliers`(df, method, k, z_threshold, cols) | MUST | T-108 |  |
| F-109 | `data\eda.py` | `compute_distribution`(series, bins) | MUST | T-109 |  |
| F-110 | `data\eda.py` | `analyze_target`(df, target_col, task) | MUST | T-110 |  |
| F-111 | `data\feature_engineer.py` | Class `InteractionTransformer` | MUST | T-111 |  |
| F-112 | `data\feature_engineer.py` | Class `GroupAggTransformer` | MUST | T-112 |  |
| F-113 | `data\feature_engineer.py` | Class `DatetimeFeatureExtractor` | MUST | T-113 |  |
| F-114 | `data\feature_engineer.py` | Class `LagRollingTransformer` | MUST | T-114 |  |
| F-115 | `data\feature_engineer.py` | Class `FeatureEngineeringConfig` | MUST | T-115 |  |
| F-116 | `data\feature_engineer.py` | `build_feature_engineering_pipeline`(config) | MUST | T-116 |  |
| F-117 | `data\leakage_detector.py` | Class `LeakagePair` | MUST | T-117 |  |
| F-118 | `data\leakage_detector.py` | Class `LeakageReport` | MUST | T-118 |  |
| F-119 | `data\leakage_detector.py` | Class `FeatureLeakageWarning` | MUST | T-119 |  |
| F-120 | `data\leakage_detector.py` | Class `FeatureLeakageReport` | MUST | T-120 |  |
| F-121 | `data\leakage_detector.py` | `compute_hat_matrix`(X) | MUST | T-121 |  |
| F-122 | `data\leakage_detector.py` | `compute_rbf_gram`(X, gamma) | MUST | T-122 |  |
| F-123 | `data\leakage_detector.py` | `compute_rf_proximity`(X, y, n_estimators, random_state) | MUST | T-123 |  |
| F-124 | `data\leakage_detector.py` | `estimate_groups`(S, n_clusters_range, method) | MUST | T-124 |  |
| F-125 | `data\leakage_detector.py` | `detect_leakage`(X, y, method, similarity_threshold, top_k, rf_n_estimators) | MUST | T-125 |  |
| F-126 | `data\leakage_detector.py` | `check_feature_leakage`(df, target_col, exclude_cols, corr_threshold_high, corr_threshold_medium, max_features_to_check) | MUST | T-126 |  |
| F-127 | `data\loader.py` | `load_file`(path, smiles_col, target_col, sqlite_query) | MUST | T-127 |  |
| F-128 | `data\loader.py` | `load_from_bytes`(content, filename) | MUST | T-128 |  |
| F-129 | `data\loader.py` | `save_dataframe`(df, path, fmt) | MUST | T-129 |  |
| F-130 | `data\loader.py` | `get_supported_extensions`() | MUST | T-130 |  |
| F-131 | `data\preprocessor.py` | Class `LogTransformer` | MUST | T-131 |  |
| F-132 | `data\preprocessor.py` | Class `SinCosTransformer` | MUST | T-132 |  |
| F-133 | `data\preprocessor.py` | Class `PreprocessConfig` | MUST | T-133 |  |
| F-134 | `data\preprocessor.py` | Class `Preprocessor` | MUST | T-134 |  |
| F-135 | `data\preprocessor.py` | `build_full_pipeline`(detection_result, model, target_col, config) | MUST | T-135 |  |
| F-136 | `data\random_projection.py` | Class `JLRandomProjection` | MUST | T-136 |  |
| F-137 | `data\random_projection.py` | `should_apply_random_projection`(n_features, n_samples, eps) | MUST | T-137 |  |
| F-138 | `data\type_detector.py` | Class `ColumnType` | MUST | T-138 |  |
| F-139 | `data\type_detector.py` | Class `ColumnInfo` | MUST | T-139 |  |
| F-140 | `data\type_detector.py` | Class `DetectionResult` | MUST | T-140 |  |
| F-141 | `data\type_detector.py` | Class `TypeDetector` | MUST | T-141 |  |
| F-142 | `doe\candidate.py` | `generate_candidate_set`(factors, max_candidates, random_seed) | MUST | T-142 |  |
| F-143 | `doe\candidate.py` | `build_model_matrix`(factors, df) | MUST | T-143 |  |
| F-144 | `doe\candidate.py` | `col_names`(factors) | MUST | T-144 |  |
| F-145 | `doe\design.py` | Class `DoEResult` | MUST | T-145 |  |
| F-146 | `doe\design.py` | Class `DoEOptimizer` | MUST | T-146 |  |
| F-147 | `doe\factor.py` | Class `FactorType` | MUST | T-147 |  |
| F-148 | `doe\factor.py` | Class `Factor` | MUST | T-148 |  |
| F-149 | `doe\orthogonal.py` | `list_oa_names`() | MUST | T-149 |  |
| F-150 | `doe\orthogonal.py` | `get_oa_info`(name) | MUST | T-150 |  |
| F-151 | `doe\orthogonal.py` | `apply_orthogonal_array`(oa_name, factors) | MUST | T-151 |  |
| F-152 | `export\base.py` | Class `BaseExporter` | MUST | T-152 |  |
| F-153 | `export\chart_bundle.py` | Class `ChartBundleExporter` | MUST | T-153 |  |
| F-154 | `export\notebook_exporter.py` | Class `NotebookExporter` | MUST | T-154 |  |
| F-155 | `export\pdf_exporter.py` | Class `PDFExporter` | MUST | T-155 |  |
| F-156 | `export\word_exporter.py` | Class `WordExporter` | MUST | T-156 |  |
| F-157 | `hsp\hsp_calculator.py` | Class `HSPCalculator` | MUST | T-157 |  |
| F-158 | `hsp\hsp_predictor.py` | Class `GroupContribution` | MUST | T-158 |  |
| F-159 | `hsp\hsp_predictor.py` | Class `HSPPredictor` | MUST | T-159 |  |
| F-160 | `interpret\shap_explainer.py` | Class `ShapConfig` | MUST | T-160 |  |
| F-161 | `interpret\shap_explainer.py` | Class `ShapResult` | MUST | T-161 |  |
| F-162 | `interpret\shap_explainer.py` | Class `ShapExplainer` | MUST | T-162 |  |
| F-163 | `interpret\sri.py` | Class `SRIResult` | MUST | T-163 |  |
| F-164 | `interpret\sri.py` | Class `SRIDecomposer` | MUST | T-164 |  |
| F-165 | `interpret\sri.py` | `plot_sri_heatmap`(sri_result, component, top_n, ax, save_path) | MUST | T-165 |  |
| F-166 | `interpret\sri.py` | `select_features_by_independence`(sri_result, top_n, threshold) | MUST | T-166 |  |
| F-167 | `llm\generator.py` | Class `LLMGeneratorError` | MUST | T-167 |  |
| F-168 | `llm\generator.py` | Class `GenerationResult` | MUST | T-168 |  |
| F-169 | `llm\generator.py` | Class `LLMDescriptorGenerator` | MUST | T-169 |  |
| F-170 | `llm\prompt_builder.py` | Class `DescriptorIntent` | MUST | T-170 |  |
| F-171 | `llm\prompt_builder.py` | `build_external_llm_prompt`(intent) | MUST | T-171 |  |
| F-172 | `llm\provider.py` | Class `LLMProviderError` | MUST | T-172 |  |
| F-173 | `llm\provider.py` | Class `LLMRequest` | MUST | T-173 |  |
| F-174 | `llm\provider.py` | Class `LLMResponse` | MUST | T-174 |  |
| F-175 | `llm\provider.py` | Class `LLMProvider` | MUST | T-175 |  |
| F-176 | `llm\provider.py` | Class `StubLLMProvider` | MUST | T-176 |  |
| F-177 | `llm\registry.py` | Class `LLMProviderRegistry` | MUST | T-177 |  |
| F-178 | `llm\reviewer.py` | Class `ReviewIssue` | MUST | T-178 |  |
| F-179 | `llm\reviewer.py` | Class `CodeReviewResult` | MUST | T-179 |  |
| F-180 | `llm\reviewer.py` | Class `LLMCodeReviewer` | MUST | T-180 |  |
| F-181 | `llm\__init__.py` | `get_llm_provider`(name) | MUST | T-181 |  |
| F-182 | `llm\__init__.py` | `register_llm_provider`(name, cls) | MUST | T-182 |  |
| F-183 | `llm\providers\hf_provider.py` | Class `DownloadProgress` | MUST | T-183 |  |
| F-184 | `llm\providers\hf_provider.py` | Class `HuggingFaceProvider` | MUST | T-184 |  |
| F-185 | `llm\providers\hf_provider.py` | `load_hf_config`() | MUST | T-185 |  |
| F-186 | `llm\providers\hf_provider.py` | `save_hf_config`(config) | MUST | T-186 |  |
| F-187 | `llm\providers\hf_provider.py` | `get_hf_token`() | MUST | T-187 |  |
| F-188 | `llm\providers\hf_provider.py` | `get_model_info`(model_id) | MUST | T-188 |  |
| F-189 | `llm\providers\hf_provider.py` | `get_download_progress`(model_id) | MUST | T-189 |  |
| F-190 | `llm\providers\hf_provider.py` | `is_model_downloaded`(model_id) | MUST | T-190 |  |
| F-191 | `llm\providers\hf_provider.py` | `download_model_async`(model_id, token, on_progress) | MUST | T-191 |  |
| F-192 | `llm\providers\hf_provider.py` | `load_model`(model_id, token) | MUST | T-192 |  |
| F-193 | `llm\providers\openai_provider.py` | Class `OpenAIProvider` | MUST | T-193 |  |
| F-194 | `mlops\mlflow_manager.py` | Class `MLflowManager` | MUST | T-194 |  |
| F-195 | `mlops\mlflow_manager.py` | Class `MLRunContext` | MUST | T-195 |  |
| F-196 | `models\automl.py` | Class `AutoMLResult` | MUST | T-196 |  |
| F-197 | `models\automl.py` | Class `AutoMLEngine` | MUST | T-197 |  |
| F-198 | `models\cv_bias_evaluator.py` | Class `CVBiasResult` | MUST | T-198 |  |
| F-199 | `models\cv_bias_evaluator.py` | `estimate_tibshirani_bias`(fold_error_curves, param_values, higher_is_better) | MUST | T-199 |  |
| F-200 | `models\cv_bias_evaluator.py` | `estimate_bbc_cv_bias`(oos_predictions, y_true, scoring_func, n_bootstrap, random_state, higher_is_better) | MUST | T-200 |  |
| F-201 | `models\cv_bias_evaluator.py` | `format_bias_report`(result) | MUST | T-201 |  |
| F-202 | `models\cv_manager.py` | Class `WalkForwardSplit` | MUST | T-202 |  |
| F-203 | `models\cv_manager.py` | Class `CVConfig` | MUST | T-203 |  |
| F-204 | `models\cv_manager.py` | `get_cv`(config) | MUST | T-204 |  |
| F-205 | `models\cv_manager.py` | `list_cv_methods`(task, requires_groups) | MUST | T-205 |  |
| F-206 | `models\cv_manager.py` | `run_cross_validation`(model, X, y, cv_config, scoring, groups, n_jobs, return_train_score, fit_params) | MUST | T-206 |  |
| F-207 | `models\factory.py` | `get_model`(model_key, task) | MUST | T-207 |  |
| F-208 | `models\factory.py` | `list_models`(task, available_only, tags) | MUST | T-208 |  |
| F-209 | `models\factory.py` | `get_default_automl_models`(task) | MUST | T-209 |  |
| F-210 | `models\factory.py` | `get_model_registry`(task) | MUST | T-210 |  |
| F-211 | `models\linear_tree.py` | Class `_Node` | MUST | T-211 |  |
| F-212 | `models\linear_tree.py` | Class `_LinearTreeCore` | MUST | T-212 |  |
| F-213 | `models\linear_tree.py` | Class `LinearTreeRegressor` | MUST | T-213 |  |
| F-214 | `models\linear_tree.py` | Class `LinearTreeClassifier` | MUST | T-214 |  |
| F-215 | `models\linear_tree.py` | Class `LinearForestRegressor` | MUST | T-215 |  |
| F-216 | `models\linear_tree.py` | Class `LinearForestClassifier` | MUST | T-216 |  |
| F-217 | `models\linear_tree.py` | Class `LinearBoostRegressor` | MUST | T-217 |  |
| F-218 | `models\linear_tree.py` | Class `LinearBoostClassifier` | MUST | T-218 |  |
| F-219 | `models\linear_tree.py` | `RidgeTreeRegressor`() | MUST | T-219 |  |
| F-220 | `models\linear_tree.py` | `RidgeTreeClassifier`() | MUST | T-220 |  |
| F-221 | `models\monotonic_kernel.py` | Class `MonotonicKernelWrapper` | MUST | T-221 |  |
| F-222 | `models\monotonic_kernel.py` | Class `MonotonicKernelClassifierWrapper` | MUST | T-222 |  |
| F-223 | `models\monotonic_kernel.py` | `is_soft_monotonic_candidate`(estimator) | MUST | T-223 |  |
| F-224 | `models\monotonic_kernel.py` | `wrap_with_soft_monotonic`(estimator, monotonic_constraints) | MUST | T-224 |  |
| F-225 | `models\monotonic_wrapper.py` | Class `MonotonicConstraintRegressor` | MUST | T-225 | [x] |
| F-226 | `models\monotonic_wrapper.py` | Class `MonotonicConstraintClassifier` | MUST | T-226 | [x] |
| F-227 | `models\monotonic_wrapper.py` | `model_monotonic_strategy`(estimator) | MUST | T-227 | [x] |
| F-228 | `models\monotonic_wrapper.py` | `wrap_monotonic`(estimator, monotonic_constraints) | MUST | T-228 | [x] |
| F-229 | `models\rgf.py` | Class `_RGFCore` | MUST | T-229 | [x] |
| F-230 | `models\rgf.py` | Class `RGFRegressor` | MUST | T-230 | [x] |
| F-231 | `models\rgf.py` | Class `RGFClassifier` | MUST | T-231 | [x] |
| F-232 | `models\search_space_generator.py` | Class `SearchParamSpec` | MUST | T-232 |  |
| F-233 | `models\search_space_generator.py` | `generate_search_space`(param_spec) | MUST | T-233 |  |
| F-234 | `models\search_space_generator.py` | `generate_grid_space`(param_specs) | MUST | T-234 |  |
| F-235 | `models\search_space_generator.py` | `generate_optuna_space`(param_specs) | MUST | T-235 |  |
| F-236 | `models\search_space_generator.py` | `generate_search_spaces`(param_specs) | MUST | T-236 |  |
| F-237 | `models\search_space_generator.py` | `generate_search_spaces_from_estimator`(estimator_cls) | MUST | T-237 |  |
| F-238 | `models\tuner.py` | Class `TunerConfig` | MUST | T-238 |  |
| F-239 | `models\tuner.py` | `tune`(model, X, y, config, groups) | MUST | T-239 |  |
| F-240 | `optim\bayesian_optimizer.py` | Class `BOConfig` | MUST | T-240 |  |
| F-241 | `optim\bayesian_optimizer.py` | Class `BayesianOptimizer` | MUST | T-241 |  |
| F-242 | `optim\bo_visualizer.py` | `plot_pca_2d`(X_existing, X_candidates, feature_names, y_existing, top_n_arrows) | MUST | T-242 |  |
| F-243 | `optim\bo_visualizer.py` | `plot_pca_3d`(X_existing, X_candidates, feature_names, y_existing, top_n_arrows) | MUST | T-243 |  |
| F-244 | `optim\bo_visualizer.py` | `plot_pareto_front`(Y_existing, Y_candidates, objective_names, directions) | MUST | T-244 |  |
| F-245 | `optim\bo_visualizer.py` | `plot_convergence`(y_history, objective) | MUST | T-245 |  |
| F-246 | `optim\constraints.py` | Class `Constraint` | MUST | T-246 |  |
| F-247 | `optim\constraints.py` | Class `RangeConstraint` | MUST | T-247 |  |
| F-248 | `optim\constraints.py` | Class `SumConstraint` | MUST | T-248 |  |
| F-249 | `optim\constraints.py` | Class `InequalityConstraint` | MUST | T-249 |  |
| F-250 | `optim\constraints.py` | Class `AtLeastNConstraint` | MUST | T-250 |  |
| F-251 | `optim\constraints.py` | Class `CustomConstraint` | MUST | T-251 |  |
| F-252 | `optim\constraints.py` | `apply_constraints`(df, constraints) | MUST | T-252 |  |
| F-253 | `optim\inverse_optimizer.py` | Class `InverseConfig` | MUST | T-253 |  |
| F-254 | `optim\inverse_optimizer.py` | Class `InverseResult` | MUST | T-254 |  |
| F-255 | `optim\inverse_optimizer.py` | `run_inverse_optimization`(predict_fn, feature_names, config, progress_callback) | MUST | T-255 |  |
| F-256 | `optim\search_space.py` | Class `VarType` | MUST | T-256 |  |
| F-257 | `optim\search_space.py` | Class `Variable` | MUST | T-257 |  |
| F-258 | `optim\search_space.py` | Class `SearchSpace` | MUST | T-258 |  |
| F-259 | `pipeline\column_selector.py` | Class `ColumnMeta` | MUST | T-259 |  |
| F-260 | `pipeline\column_selector.py` | Class `ColumnSelectorWrapper` | MUST | T-260 |  |
| F-261 | `pipeline\col_preprocessor.py` | Class `ColPreprocessConfig` | MUST | T-261 |  |
| F-262 | `pipeline\col_preprocessor.py` | Class `ColPreprocessor` | MUST | T-262 |  |
| F-263 | `pipeline\feature_generator.py` | Class `FeatureGenConfig` | MUST | T-263 |  |
| F-264 | `pipeline\feature_generator.py` | Class `FeatureGenerator` | MUST | T-264 |  |
| F-265 | `pipeline\feature_selector.py` | Class `FeatureSelectorConfig` | MUST | T-265 |  |
| F-266 | `pipeline\feature_selector.py` | Class `FeatureSelector` | MUST | T-266 |  |
| F-267 | `pipeline\feature_selector.py` | Class `_GroupLassoSelector` | MUST | T-267 |  |
| F-268 | `pipeline\pipeline_builder.py` | Class `PipelineConfig` | MUST | T-268 |  |
| F-269 | `pipeline\pipeline_builder.py` | `build_pipeline`(config) | MUST | T-269 |  |
| F-270 | `pipeline\pipeline_builder.py` | `apply_monotonic_constraints`(estimator, column_meta, feature_names) | MUST | T-270 |  |
| F-271 | `pipeline\pipeline_builder.py` | `extract_group_array`(column_meta, feature_names) | MUST | T-271 |  |
| F-272 | `pipeline\pipeline_grid.py` | Class `PipelineCombination` | MUST | T-272 |  |
| F-273 | `pipeline\pipeline_grid.py` | Class `PipelineGridConfig` | MUST | T-273 |  |
| F-274 | `pipeline\pipeline_grid.py` | `generate_pipeline_grid`(grid_config, max_combinations) | MUST | T-274 |  |
| F-275 | `pipeline\pipeline_grid.py` | `count_combinations`(grid_config) | MUST | T-275 |  |
| F-276 | `session\version_manager.py` | Class `ExperimentRecord` | MUST | T-276 |  |
| F-277 | `session\version_manager.py` | Class `VersionManager` | MUST | T-277 |  |
| F-278 | `ui\param_schema.py` | Class `ParamSpec` | MUST | T-278 |  |
| F-279 | `ui\param_schema.py` | `introspect_params`(cls) | MUST | T-279 |  |
| F-280 | `ui\param_schema.py` | `introspect_adapter`(adapter_instance) | MUST | T-280 |  |
| F-281 | `ui\param_schema.py` | `introspect_adapter_class`(adapter_cls) | MUST | T-281 |  |
| F-282 | `ui\param_schema.py` | `apply_params`(specs, user_values) | MUST | T-282 |  |
| F-283 | `ui\param_schema.py` | `get_basic_specs`(specs) | MUST | T-283 |  |
| F-284 | `ui\param_schema.py` | `get_advanced_specs`(specs) | MUST | T-284 |  |
| F-285 | `utils\config.py` | Class `AppConfig` | MUST | T-285 |  |
| F-286 | `utils\cv_recommender.py` | Class `CVRecommendation` | MUST | T-286 |  |
| F-287 | `utils\cv_recommender.py` | `recommend_cv_strategy`(X, y, metadata) | MUST | T-287 |  |
| F-288 | `utils\optional_import.py` | `safe_import`(module_name, alias) | MUST | T-288 |  |
| F-289 | `utils\optional_import.py` | `is_available`(name) | MUST | T-289 |  |
| F-290 | `utils\optional_import.py` | `require`(name, feature) | MUST | T-290 |  |
| F-291 | `utils\optional_import.py` | `get_availability_report`() | MUST | T-291 |  |
| F-292 | `utils\optional_import.py` | `probe_all_optional_libraries`() | MUST | T-292 |  |
| F-293 | `main.py` | `main_page`() | MUST | T-293 |  |
| F-294 | `main.py` | `help_page`() | MUST | T-294 |  |
| F-295 | `main.py` | `help_descriptors_page`() | MUST | T-295 |  |
| F-296 | `components\analysis_runner.py` | Class `AnalysisCancelled` | MUST | T-296 |  |
| F-297 | `components\auto_params_ui.py` | `render_param_editor`(specs, title) | MUST | T-297 |  |
| F-298 | `components\auto_params_ui.py` | `render_model_param_editor`(model_cls, title) | MUST | T-298 |  |
| F-299 | `components\auto_params_ui.py` | `render_adapter_param_editor`(adapter_cls, title) | MUST | T-299 |  |
| F-300 | `components\batch_predict_tab.py` | `render_batch_predict_tab`(state) | MUST | T-300 |  |
| F-301 | `components\bayesian_opt_ui.py` | `render_bayesian_opt_ui`(state) | MUST | T-301 |  |
| F-302 | `components\column_meta_editor.py` | `render_column_meta_editor`(state, df) | MUST | T-302 |  |
| F-303 | `components\column_meta_editor.py` | `build_column_meta_dict`(state) | MUST | T-303 |  |
| F-304 | `components\column_meta_editor.py` | `extract_monotonic_from_column_meta`(state) | MUST | T-304 |  |
| F-305 | `components\constraint_ui_helpers.py` | `ui_constraints_to_backend`(constraint_items) | MUST | T-305 |  |
| F-306 | `components\constraint_ui_helpers.py` | `validate_constraints`(constraint_items, df) | MUST | T-306 |  |
| F-307 | `components\constraint_ui_helpers.py` | `describe_constraint`(item) | MUST | T-307 |  |
| F-308 | `components\constraint_ui_helpers.py` | `save_template`(name, constraint_items, tags) | MUST | T-308 |  |
| F-309 | `components\constraint_ui_helpers.py` | `load_template`(path) | MUST | T-309 |  |
| F-310 | `components\constraint_ui_helpers.py` | `list_templates`() | MUST | T-310 |  |
| F-311 | `components\constraint_ui_helpers.py` | `delete_template`(path) | MUST | T-311 |  |
| F-312 | `components\constraint_ui_helpers.py` | `friendly_error`(error) | MUST | T-312 |  |
| F-313 | `components\constraint_ui_helpers.py` | `get_column_stats`(df, col) | MUST | T-313 |  |
| F-314 | `components\cv_config_ui.py` | `render_cv_config`(state) | MUST | T-314 |  |
| F-315 | `components\data_tab.py` | `render_data_tab`(state) | MUST | T-315 |  |
| F-316 | `components\descriptor_catalog.py` | `get_rdkit_catalog`() | MUST | T-316 |  |
| F-317 | `components\descriptor_catalog.py` | `get_xtb_catalog`() | MUST | T-317 |  |
| F-318 | `components\descriptor_catalog.py` | `get_group_contrib_catalog`() | MUST | T-318 |  |
| F-319 | `components\descriptor_catalog.py` | `get_skfp_catalog`() | MUST | T-319 |  |
| F-320 | `components\descriptor_catalog.py` | `get_mordred_catalog`() | MUST | T-320 |  |
| F-321 | `components\descriptor_catalog.py` | `get_molfeat_catalog`() | MUST | T-321 |  |
| F-322 | `components\descriptor_catalog.py` | `get_cosmo_catalog`() | MUST | T-322 |  |
| F-323 | `components\descriptor_catalog.py` | `get_descriptastorus_catalog`() | MUST | T-323 |  |
| F-324 | `components\descriptor_catalog.py` | `get_padel_catalog`() | MUST | T-324 |  |
| F-325 | `components\descriptor_catalog.py` | `get_catalog`(engine_name) | MUST | T-325 |  |
| F-326 | `components\descriptor_help_page.py` | `render_descriptor_help`() | MUST | T-326 |  |
| F-327 | `components\descriptor_plugins_ui.py` | `render_descriptor_plugins`(state) | MUST | T-327 |  |
| F-328 | `components\descriptor_selector_dialog.py` | `open_descriptor_detail_dialog`(engine_name, state) | MUST | T-328 |  |
| F-329 | `components\descriptor_selector_dialog.py` | `render_selected_descriptors_panel`(state) | MUST | T-329 |  |
| F-330 | `components\descriptor_selector_dialog.py` | `render_descriptor_sets_panel`(state) | MUST | T-330 |  |
| F-331 | `components\descriptor_status_bar.py` | `render_descriptor_status_bar`(state) | MUST | T-331 |  |
| F-332 | `components\dialog_manager.py` | Class `StateSnapshot` | MUST | T-332 |  |
| F-333 | `components\dialog_manager.py` | `create_settings_dialog`() | MUST | T-333 |  |
| F-334 | `components\dialog_manager.py` | `render_settings_summary`() | MUST | T-334 |  |
| F-335 | `components\doe_tab.py` | `render_doe_tab`(app_state) | MUST | T-335 |  |
| F-336 | `components\eda_panel.py` | `render_eda_panel`(state) | MUST | T-336 |  |
| F-337 | `components\estimator_config_dialog.py` | Class `EstimatorConfig` | MUST | T-337 |  |
| F-338 | `components\estimator_config_dialog.py` | Class `EstimatorConfigDialog` | MUST | T-338 |  |
| F-339 | `components\estimator_config_dialog.py` | `render_model_config_panel`(model_entries, state) | MUST | T-339 |  |
| F-340 | `components\feature_set_manager.py` | `render_feature_set_manager`(state) | MUST | T-340 |  |
| F-341 | `components\internal_llm_ui.py` | `render_internal_llm_tab`(state) | MUST | T-341 |  |
| F-342 | `components\interpretation_panel.py` | `render_interpretation_panel`(ar, state) | MUST | T-342 |  |
| F-343 | `components\inverse_analysis_tab.py` | `render_inverse_analysis_tab`(state) | MUST | T-343 |  |
| F-344 | `components\leakage_check_ui.py` | `render_leakage_check_panel`(state) | MUST | T-344 |  |
| F-345 | `components\pipeline_config_ui.py` | `render_pipeline_config`(state) | MUST | T-345 |  |
| F-346 | `components\post_analysis_config.py` | `render_post_analysis_config`(state) | MUST | T-346 |  |
| F-347 | `components\report_generator.py` | `render_report_tab`(state) | MUST | T-347 |  |
| F-348 | `components\results_tab.py` | `render_results_tab`(state) | MUST | T-348 |  |
| F-349 | `components\settings_checker.py` | `render_settings_checker`(state) | MUST | T-349 |  |
| F-350 | `components\smiles_hover.py` | `smiles_to_svg_b64`(smiles, width, height) | MUST | T-350 |  |
| F-351 | `components\smiles_hover.py` | `smiles_to_png_b64`(smiles, width, height) | MUST | T-351 |  |
| F-352 | `components\smiles_hover.py` | `render_smiles_table`(df, smiles_col, max_rows, img_size, height) | MUST | T-352 |  |
| F-353 | `components\smiles_hover.py` | `add_smiles_hover_to_plotly`(fig, smiles_list, label_list, img_size) | MUST | T-353 |  |
| F-354 | `components\tuning_tab.py` | `render_tuning_tab`(state) | MUST | T-354 |  |
| F-355 | `pages\experiment_comparison.py` | `render_experiment_comparison`(state) | MUST | T-355 |  |
| F-356 | `pages\export_panel.py` | `render_export_panel`(state) | MUST | T-356 |  |
| F-357 | `pages\model_manager.py` | `render_model_manager`() | MUST | T-357 |  |
| F-358 | `conftest.py` | `random_seed`() | MUST | T-358 |  |
| F-359 | `conftest.py` | `small_regression_df`() | MUST | T-359 |  |
| F-360 | `conftest.py` | `small_classification_df`() | MUST | T-360 |  |
| F-361 | `debug_adapters.py` | `test_adapter`(module_path, class_name, kwargs, status_tag) | MUST | T-361 |  |
| F-362 | `debug_adapters.py` | `main`() | MUST | T-362 |  |
| F-363 | `test_automl_extra.py` | Class `TestAutoMLResult` | MUST | T-363 |  |
| F-364 | `test_automl_extra.py` | Class `TestAutoMLEngine` | MUST | T-364 |  |
| F-365 | `test_automl_integration.py` | Class `TestAutoMLInit` | MUST | T-365 |  |
| F-366 | `test_automl_integration.py` | Class `TestAutoMLRegression` | MUST | T-366 |  |
| F-367 | `test_automl_integration.py` | Class `TestAutoMLClassification` | MUST | T-367 |  |
| F-368 | `test_automl_integration.py` | Class `TestAutoMLResult` | MUST | T-368 |  |
| F-369 | `test_automl_integration.py` | `regression_df`() | MUST | T-369 |  |
| F-370 | `test_automl_integration.py` | `classification_df`() | MUST | T-370 |  |
| F-371 | `test_base_and_feature_gen.py` | Class `TestDescriptorResult` | MUST | T-371 |  |
| F-372 | `test_base_and_feature_gen.py` | Class `TestDescriptorMetadata` | MUST | T-372 |  |
| F-373 | `test_base_and_feature_gen.py` | Class `DummyAdapter` | MUST | T-373 |  |
| F-374 | `test_base_and_feature_gen.py` | Class `UnavailableAdapter` | MUST | T-374 |  |
| F-375 | `test_base_and_feature_gen.py` | Class `TestBaseChemAdapter` | MUST | T-375 |  |
| F-376 | `test_base_and_feature_gen.py` | Class `TestFeatureGenConfig` | MUST | T-376 |  |
| F-377 | `test_base_and_feature_gen.py` | Class `TestFeatureGenerator` | MUST | T-377 |  |
| F-378 | `test_bayesian_opt.py` | Class `TestVariable` | MUST | T-378 |  |
| F-379 | `test_bayesian_opt.py` | Class `TestSearchSpace` | MUST | T-379 |  |
| F-380 | `test_bayesian_opt.py` | Class `TestConstraints` | MUST | T-380 |  |
| F-381 | `test_bayesian_opt.py` | Class `TestBayesianOptimizer` | MUST | T-381 |  |
| F-382 | `test_bayesian_optimizer_comprehensive.py` | Class `TestBOConfig` | MUST | T-382 |  |
| F-383 | `test_bayesian_optimizer_comprehensive.py` | Class `TestBayesianOptimizerFit` | MUST | T-383 |  |
| F-384 | `test_bayesian_optimizer_comprehensive.py` | Class `TestAcquisitionFunctions` | MUST | T-384 |  |
| F-385 | `test_bayesian_optimizer_comprehensive.py` | Class `TestBatchStrategies` | MUST | T-385 |  |
| F-386 | `test_bayesian_optimizer_comprehensive.py` | Class `TestPredict` | MUST | T-386 |  |
| F-387 | `test_bayesian_optimizer_comprehensive.py` | Class `TestKernelTypes` | MUST | T-387 |  |
| F-388 | `test_bayesian_optimizer_comprehensive.py` | Class `TestGPInfo` | MUST | T-388 |  |
| F-389 | `test_bayesian_optimizer_comprehensive.py` | Class `TestSuggestDataFrame` | MUST | T-389 |  |
| F-390 | `test_bayesian_optimizer_comprehensive.py` | Class `TestMaximinSelect` | MUST | T-390 |  |
| F-391 | `test_bayesian_optimizer_comprehensive.py` | Class `TestMultiObjective` | MUST | T-391 |  |
| F-392 | `test_bayesian_optimizer_comprehensive.py` | `reg_data`() | MUST | T-392 |  |
| F-393 | `test_bayesian_optimizer_comprehensive.py` | `candidates`() | MUST | T-393 |  |
| F-394 | `test_bayesian_optimizer_extra.py` | Class `TestBOConfig` | MUST | T-394 |  |
| F-395 | `test_bayesian_optimizer_extra.py` | Class `TestBayesianOptimizerBasic` | MUST | T-395 |  |
| F-396 | `test_bayesian_optimizer_extra.py` | Class `TestKernelBuilding` | MUST | T-396 |  |
| F-397 | `test_bayesian_optimizer_extra.py` | Class `TestAcquisitionFunctions` | MUST | T-397 |  |
| F-398 | `test_bayesian_optimizer_extra.py` | Class `TestBatchStrategies` | MUST | T-398 |  |
| F-399 | `test_bayesian_optimizer_extra.py` | Class `TestMultiObjective` | MUST | T-399 |  |
| F-400 | `test_bayesian_optimizer_extra.py` | Class `TestMaximinSelect` | MUST | T-400 |  |
| F-401 | `test_benchmark.py` | Class `TestModelScore` | MUST | T-401 |  |
| F-402 | `test_benchmark.py` | Class `TestBenchmarkResult` | MUST | T-402 |  |
| F-403 | `test_benchmark.py` | Class `TestEvaluateRegression` | MUST | T-403 |  |
| F-404 | `test_benchmark.py` | Class `TestEvaluateClassification` | MUST | T-404 |  |
| F-405 | `test_benchmark.py` | Class `TestComputeLearningCurve` | MUST | T-405 |  |
| F-406 | `test_benchmark.py` | Class `TestBenchmarkModels` | MUST | T-406 |  |
| F-407 | `test_benchmark.py` | `reg_data`() | MUST | T-407 |  |
| F-408 | `test_benchmark.py` | `cls_data`() | MUST | T-408 |  |
| F-409 | `test_benchmark_datasets_extra.py` | Class `TestListBenchmarkDatasets` | MUST | T-409 |  |
| F-410 | `test_benchmark_datasets_extra.py` | Class `TestBenchmarkURLs` | MUST | T-410 |  |
| F-411 | `test_benchmark_datasets_extra.py` | Class `TestLoadBenchmark` | MUST | T-411 |  |
| F-412 | `test_benchmark_extra.py` | Class `TestModelScore` | MUST | T-412 |  |
| F-413 | `test_benchmark_extra.py` | Class `TestBenchmarkResult` | MUST | T-413 |  |
| F-414 | `test_benchmark_extra.py` | Class `TestEvaluateRegression` | MUST | T-414 |  |
| F-415 | `test_benchmark_extra.py` | Class `TestEvaluateClassification` | MUST | T-415 |  |
| F-416 | `test_benchmark_extra.py` | Class `TestBenchmarkModels` | MUST | T-416 |  |
| F-417 | `test_bo_visualizer.py` | Class `TestPlotPCA2D` | MUST | T-417 |  |
| F-418 | `test_bo_visualizer.py` | Class `TestPlotPCA3D` | MUST | T-418 |  |
| F-419 | `test_bo_visualizer.py` | Class `TestPlotParetoFront` | MUST | T-419 |  |
| F-420 | `test_bo_visualizer.py` | Class `TestParetoEfficient` | MUST | T-420 |  |
| F-421 | `test_bo_visualizer.py` | Class `TestPlotConvergence` | MUST | T-421 |  |
| F-422 | `test_bo_visualizer_comprehensive.py` | Class `TestPlotPCA2D` | MUST | T-422 |  |
| F-423 | `test_bo_visualizer_comprehensive.py` | Class `TestPlotPCA3D` | MUST | T-423 |  |
| F-424 | `test_bo_visualizer_comprehensive.py` | Class `TestParetoFront` | MUST | T-424 |  |
| F-425 | `test_bo_visualizer_comprehensive.py` | Class `TestParetoEfficient` | MUST | T-425 |  |
| F-426 | `test_bo_visualizer_comprehensive.py` | Class `TestConvergence` | MUST | T-426 |  |
| F-427 | `test_bo_visualizer_comprehensive.py` | `data_5d`() | MUST | T-427 |  |
| F-428 | `test_charge_config.py` | Class `TestMoleculeChargeConfigDefaults` | MUST | T-428 |  |
| F-429 | `test_charge_config.py` | Class `TestMoleculeChargeConfigValidation` | MUST | T-429 |  |
| F-430 | `test_charge_config.py` | Class `TestMoleculeChargeConfigPresets` | MUST | T-430 |  |
| F-431 | `test_charge_config.py` | Class `TestChargeConfigStore` | MUST | T-431 |  |
| F-432 | `test_charge_config.py` | Class `TestReadSmilesFormalCharge` | MUST | T-432 |  |
| F-433 | `test_charge_config.py` | Class `TestApplyProtonation` | MUST | T-433 |  |
| F-434 | `test_charge_config.py` | Class `TestApplyProtonationBatch` | MUST | T-434 |  |
| F-435 | `test_charge_config.py` | Class `TestGetProtonationStateInfo` | MUST | T-435 |  |
| F-436 | `test_charge_config.py` | Class `TestGetUnipkaModel` | MUST | T-436 |  |
| F-437 | `test_charge_config.py` | Class `TestXTBArgsIntegration` | MUST | T-437 |  |
| F-438 | `test_charge_config.py` | `default_cfg`() | MUST | T-438 |  |
| F-439 | `test_charge_config.py` | `store`() | MUST | T-439 |  |
| F-440 | `test_charge_config_extra.py` | Class `TestMoleculeChargeConfig` | MUST | T-440 |  |
| F-441 | `test_charge_config_extra.py` | Class `TestChargeConfigStore` | MUST | T-441 |  |
| F-442 | `test_charge_config_extra.py` | Class `TestReadSmilesFormalCharge` | MUST | T-442 |  |
| F-443 | `test_charge_extended.py` | Class `TestParseXtbOutput` | MUST | T-443 |  |
| F-444 | `test_charge_extended.py` | Class `TestSmilesToXyz` | MUST | T-444 |  |
| F-445 | `test_charge_extended.py` | Class `TestRDKitAdapterGasteigerCharges` | MUST | T-445 |  |
| F-446 | `test_charge_extended.py` | Class `TestChargeConfigStoreEdgeCases` | MUST | T-446 |  |
| F-447 | `test_charge_extended.py` | Class `TestMoleculeChargeConfigEdgeCases` | MUST | T-447 |  |
| F-448 | `test_charge_extended.py` | Class `TestReadSmilesFormalChargeEdgeCases` | MUST | T-448 |  |
| F-449 | `test_charge_extended.py` | Class `TestProtonationMaxModes` | MUST | T-449 |  |
| F-450 | `test_charge_extended.py` | Class `TestXTBAdapterComputeMock` | MUST | T-450 |  |
| F-451 | `test_chem.py` | `rdkit_adapter`() | MUST | T-451 |  |
| F-452 | `test_chem.py` | `test_rdkit_adapter_name`(rdkit_adapter) | MUST | T-452 |  |
| F-453 | `test_chem.py` | `test_rdkit_adapter_availability`(rdkit_adapter) | MUST | T-453 |  |
| F-454 | `test_chem.py` | `test_compute_descriptors`(rdkit_adapter) | MUST | T-454 |  |
| F-455 | `test_chem.py` | `test_compute_with_invalid_smiles`(rdkit_adapter) | MUST | T-455 |  |
| F-456 | `test_chem.py` | `test_get_descriptor_names`(rdkit_adapter) | MUST | T-456 |  |
| F-457 | `test_chem.py` | `test_descriptor_metadata_rdkit`(rdkit_adapter) | MUST | T-457 |  |
| F-458 | `test_chem.py` | `test_descriptor_metadata_mordred`() | MUST | T-458 |  |
| F-459 | `test_chem.py` | `test_mordred_adapter`() | MUST | T-459 |  |
| F-460 | `test_chem.py` | `test_stub_adapters`() | MUST | T-460 |  |
| F-461 | `test_chem.py` | `test_psmiles_adapter`() | MUST | T-461 |  |
| F-462 | `test_chemprop_adapter.py` | Class `TestChempropAdapter` | MUST | T-462 |  |
| F-463 | `test_chem_base_extra.py` | Class `TestDescriptorResult` | MUST | T-463 |  |
| F-464 | `test_chem_base_extra.py` | Class `TestDescriptorMetadata` | MUST | T-464 |  |
| F-465 | `test_chem_base_extra.py` | Class `_DummyAdapter` | MUST | T-465 |  |
| F-466 | `test_chem_base_extra.py` | Class `_UnavailableAdapter` | MUST | T-466 |  |
| F-467 | `test_chem_base_extra.py` | Class `TestBaseChemAdapter` | MUST | T-467 |  |
| F-468 | `test_chem_init.py` | Class `TestChemInitSafeImport` | MUST | T-468 |  |
| F-469 | `test_chem_init_extra.py` | Class `TestMakeUnavailableAdapter` | MUST | T-469 |  |
| F-470 | `test_chem_init_extra.py` | Class `TestAdapterRegistry` | MUST | T-470 |  |
| F-471 | `test_chem_init_extra.py` | Class `TestGetAvailableAdapters` | MUST | T-471 |  |
| F-472 | `test_chem_init_extra.py` | Class `TestAllExports` | MUST | T-472 |  |
| F-473 | `test_column_meta_integration.py` | Class `TestColumnMetaExtended` | MUST | T-473 |  |
| F-474 | `test_column_meta_integration.py` | Class `TestColumnMetaEditorUtils` | MUST | T-474 |  |
| F-475 | `test_column_meta_integration.py` | Class `TestFeatureSelectorFixed` | MUST | T-475 |  |
| F-476 | `test_column_meta_integration.py` | Class `TestAutoMLEngineColumnMeta` | MUST | T-476 |  |
| F-477 | `test_column_selector_extra.py` | Class `TestColumnMeta` | MUST | T-477 |  |
| F-478 | `test_column_selector_extra.py` | Class `TestColumnSelectorAll` | MUST | T-478 |  |
| F-479 | `test_column_selector_extra.py` | Class `TestColumnSelectorInclude` | MUST | T-479 |  |
| F-480 | `test_column_selector_extra.py` | Class `TestColumnSelectorExclude` | MUST | T-480 |  |
| F-481 | `test_column_selector_extra.py` | Class `TestColumnSelectorErrors` | MUST | T-481 |  |
| F-482 | `test_column_selector_extra.py` | Class `TestColumnSelectorMeta` | MUST | T-482 |  |
| F-483 | `test_col_preprocessor_comprehensive.py` | Class `TestColPreprocessConfig` | MUST | T-483 |  |
| F-484 | `test_col_preprocessor_comprehensive.py` | Class `TestColPreprocessor` | MUST | T-484 |  |
| F-485 | `test_col_preprocessor_comprehensive.py` | `mixed_df`() | MUST | T-485 |  |
| F-486 | `test_col_preprocessor_extra.py` | Class `TestColPreprocessorBasic` | MUST | T-486 |  |
| F-487 | `test_col_preprocessor_extra.py` | Class `TestScalers` | MUST | T-487 |  |
| F-488 | `test_col_preprocessor_extra.py` | Class `TestImputers` | MUST | T-488 |  |
| F-489 | `test_col_preprocessor_extra.py` | Class `TestEncoders` | MUST | T-489 |  |
| F-490 | `test_col_preprocessor_extra.py` | Class `TestBinary` | MUST | T-490 |  |
| F-491 | `test_col_preprocessor_extra.py` | Class `TestOverrideTypes` | MUST | T-491 |  |
| F-492 | `test_col_preprocessor_extra.py` | Class `TestEdgeCases` | MUST | T-492 |  |
| F-493 | `test_config_extra.py` | Class `TestConstants` | MUST | T-493 |  |
| F-494 | `test_config_extra.py` | Class `TestAppConfig` | MUST | T-494 |  |
| F-495 | `test_constraints_extra.py` | Class `TestRangeConstraint` | MUST | T-495 |  |
| F-496 | `test_constraints_extra.py` | Class `TestSumConstraint` | MUST | T-496 |  |
| F-497 | `test_constraints_extra.py` | Class `TestInequalityConstraint` | MUST | T-497 |  |
| F-498 | `test_constraints_extra.py` | Class `TestAtLeastNConstraint` | MUST | T-498 |  |
| F-499 | `test_constraints_extra.py` | Class `TestCustomConstraint` | MUST | T-499 |  |
| F-500 | `test_constraints_extra.py` | Class `TestApplyConstraints` | MUST | T-500 |  |
| F-501 | `test_constraint_ui_helpers.py` | Class `TestConstants` | MUST | T-501 |  |
| F-502 | `test_constraint_ui_helpers.py` | Class `TestUIToBackendConversion` | MUST | T-502 |  |
| F-503 | `test_constraint_ui_helpers.py` | Class `TestDescribeConstraint` | MUST | T-503 |  |
| F-504 | `test_constraint_ui_helpers.py` | Class `TestValidateConstraints` | MUST | T-504 |  |
| F-505 | `test_constraint_ui_helpers.py` | Class `TestTemplateIO` | MUST | T-505 |  |
| F-506 | `test_constraint_ui_helpers.py` | Class `TestFriendlyError` | MUST | T-506 |  |
| F-507 | `test_constraint_ui_helpers.py` | Class `TestGetColumnStats` | MUST | T-507 |  |
| F-508 | `test_cosmo_adapter.py` | Class `TestCosmoAdapterProperties` | MUST | T-508 |  |
| F-509 | `test_cosmo_adapter.py` | Class `TestCosmoAdapterAvailability` | MUST | T-509 |  |
| F-510 | `test_cosmo_adapter.py` | Class `TestCosmoAdapterComputeNoCosmiFiles` | MUST | T-510 |  |
| F-511 | `test_cosmo_adapter.py` | Class `TestCosmoAdapterComputeWithCosmiFiles` | MUST | T-511 |  |
| F-512 | `test_cosmo_adapter.py` | Class `TestCosmoDescriptorConstants` | MUST | T-512 |  |
| F-513 | `test_coverage_gaps_batch3.py` | Class `TestEdaCoverageGaps` | MUST | T-513 |  |
| F-514 | `test_coverage_gaps_batch3.py` | Class `TestDataCleanerCoverageGaps` | MUST | T-514 |  |
| F-515 | `test_coverage_gaps_batch3.py` | Class `TestFeatureEngineerCoverageGaps` | MUST | T-515 |  |
| F-516 | `test_coverage_gaps_batch3.py` | Class `TestRgfCoverageGaps` | MUST | T-516 |  |
| F-517 | `test_coverage_gaps_batch3.py` | Class `TestConstraintsCoverageGaps` | MUST | T-517 |  |
| F-518 | `test_coverage_gaps_batch3.py` | Class `TestSearchSpaceCoverageGaps` | MUST | T-518 |  |
| F-519 | `test_coverage_gaps_batch3.py` | Class `TestBayesianOptimizerCoverageGaps` | MUST | T-519 |  |
| F-520 | `test_coverage_gaps_batch3.py` | Class `TestBOVisualizerCoverageGaps` | MUST | T-520 |  |
| F-521 | `test_coverage_gaps_batch3.py` | Class `TestBenchmarkCoverageGaps` | MUST | T-521 |  |
| F-522 | `test_coverage_gaps_batch3.py` | Class `TestCVBiasEvaluatorCoverageGaps` | MUST | T-522 |  |
| F-523 | `test_coverage_gaps_batch3.py` | Class `TestColumnSelectorCoverageGaps` | MUST | T-523 |  |
| F-524 | `test_coverage_gaps_batch3.py` | Class `TestRecommenderCoverageGaps` | MUST | T-524 |  |
| F-525 | `test_coverage_gaps_batch3.py` | Class `TestChargeConfigCoverageGaps` | MUST | T-525 |  |
| F-526 | `test_coverage_gaps_batch3.py` | Class `TestGroupContribCoverageGaps` | MUST | T-526 |  |
| F-527 | `test_coverage_gaps_batch3.py` | Class `TestSRICoverageGaps` | MUST | T-527 |  |
| F-528 | `test_coverage_gaps_batch3.py` | Class `TestOptionalImportCoverageGaps` | MUST | T-528 |  |
| F-529 | `test_coverage_gaps_batch3.py` | Class `TestMlflowCoverageGaps` | MUST | T-529 |  |
| F-530 | `test_cv_bias_evaluator.py` | Class `TestCVBiasResult` | MUST | T-530 |  |
| F-531 | `test_cv_bias_evaluator.py` | Class `TestTibshiraniBias` | MUST | T-531 |  |
| F-532 | `test_cv_bias_evaluator.py` | Class `TestBBCCV` | MUST | T-532 |  |
| F-533 | `test_cv_bias_evaluator.py` | Class `TestFormatBiasReport` | MUST | T-533 |  |
| F-534 | `test_cv_bias_evaluator_extra.py` | Class `TestCVBiasResult` | MUST | T-534 |  |
| F-535 | `test_cv_bias_evaluator_extra.py` | Class `TestTibshiraniBias` | MUST | T-535 |  |
| F-536 | `test_cv_bias_evaluator_extra.py` | Class `TestBBCCVBias` | MUST | T-536 |  |
| F-537 | `test_cv_bias_evaluator_extra.py` | Class `TestFormatBiasReport` | MUST | T-537 |  |
| F-538 | `test_cv_dynamic_extra.py` | `test_dynamic_cv_lookup`() | MUST | T-538 |  |
| F-539 | `test_cv_dynamic_extra.py` | `test_dynamic_cv_extra_params`() | MUST | T-539 |  |
| F-540 | `test_cv_dynamic_extra.py` | `test_invalid_cv_param_warning`(caplog) | MUST | T-540 |  |
| F-541 | `test_cv_manager_comprehensive.py` | Class `TestWalkForwardSplit` | MUST | T-541 |  |
| F-542 | `test_cv_manager_comprehensive.py` | Class `TestCVConfig` | MUST | T-542 |  |
| F-543 | `test_cv_manager_comprehensive.py` | Class `TestGetCV` | MUST | T-543 |  |
| F-544 | `test_cv_manager_comprehensive.py` | Class `TestListCVMethods` | MUST | T-544 |  |
| F-545 | `test_cv_manager_comprehensive.py` | Class `TestRunCrossValidation` | MUST | T-545 |  |
| F-546 | `test_cv_manager_extra.py` | Class `TestWalkForwardSplit` | MUST | T-546 |  |
| F-547 | `test_cv_manager_extra.py` | Class `TestGetCVClass` | MUST | T-547 |  |
| F-548 | `test_cv_manager_extra.py` | Class `TestCVConfig` | MUST | T-548 |  |
| F-549 | `test_cv_manager_extra.py` | Class `TestListCVMethods` | MUST | T-549 |  |
| F-550 | `test_cv_manager_extra.py` | Class `TestRunCV` | MUST | T-550 |  |
| F-551 | `test_cv_manager_new.py` | `test_cv_aliases`() | MUST | T-551 |  |
| F-552 | `test_cv_manager_new.py` | `test_type_conversion`() | MUST | T-552 |  |
| F-553 | `test_cv_manager_new.py` | `test_bool_conversion`() | MUST | T-553 |  |
| F-554 | `test_cv_recommender.py` | Class `TestNormalRegression` | MUST | T-554 |  |
| F-555 | `test_cv_recommender.py` | Class `TestTimeseriesByName` | MUST | T-555 |  |
| F-556 | `test_cv_recommender.py` | Class `TestTimeseriesMonotonic` | MUST | T-556 |  |
| F-557 | `test_cv_recommender.py` | Class `TestGroupedData` | MUST | T-557 |  |
| F-558 | `test_cv_recommender.py` | Class `TestImbalancedClassification` | MUST | T-558 |  |
| F-559 | `test_cv_recommender.py` | Class `TestVerySmallData` | MUST | T-559 |  |
| F-560 | `test_cv_recommender.py` | Class `TestSmallData` | MUST | T-560 |  |
| F-561 | `test_cv_recommender.py` | Class `TestDetectTimeseries` | MUST | T-561 |  |
| F-562 | `test_cv_recommender.py` | Class `TestDetectImbalance` | MUST | T-562 |  |
| F-563 | `test_cv_recommender.py` | Class `TestAssessSampleSize` | MUST | T-563 |  |
| F-564 | `test_cv_recommender.py` | Class `TestNumpyInput` | MUST | T-564 |  |
| F-565 | `test_cv_recommender.py` | `normal_regression_data`() | MUST | T-565 |  |
| F-566 | `test_cv_recommender.py` | `timeseries_data_by_name`() | MUST | T-566 |  |
| F-567 | `test_cv_recommender.py` | `timeseries_data_monotonic`() | MUST | T-567 |  |
| F-568 | `test_cv_recommender.py` | `grouped_data`() | MUST | T-568 |  |
| F-569 | `test_cv_recommender.py` | `imbalanced_classification_data`() | MUST | T-569 |  |
| F-570 | `test_cv_recommender.py` | `very_small_data`() | MUST | T-570 |  |
| F-571 | `test_cv_recommender.py` | `small_data`() | MUST | T-571 |  |
| F-572 | `test_cv_walkforward.py` | Class `TestWalkForwardSplit` | MUST | T-572 |  |
| F-573 | `test_cv_walkforward.py` | Class `TestCVConfig` | MUST | T-573 |  |
| F-574 | `test_cv_walkforward.py` | Class `TestListCVMethods` | MUST | T-574 |  |
| F-575 | `test_cv_walkforward.py` | Class `TestRunCrossValidation` | MUST | T-575 |  |
| F-576 | `test_data.py` | Class `TestTypeDetector` | MUST | T-576 |  |
| F-577 | `test_data.py` | Class `TestLoader` | MUST | T-577 |  |
| F-578 | `test_data.py` | Class `TestPreprocessor` | MUST | T-578 |  |
| F-579 | `test_data.py` | `sample_df`() | MUST | T-579 |  |
| F-580 | `test_data.py` | `csv_file`(tmp_path) | MUST | T-580 |  |
| F-581 | `test_data_cleaner.py` | Class `TestDropColumns` | MUST | T-581 |  |
| F-582 | `test_data_cleaner.py` | Class `TestDropRowsWithMissing` | MUST | T-582 |  |
| F-583 | `test_data_cleaner.py` | Class `TestRemoveConstantColumns` | MUST | T-583 |  |
| F-584 | `test_data_cleaner.py` | Class `TestClipOutliers` | MUST | T-584 |  |
| F-585 | `test_data_cleaner.py` | Class `TestRemoveDuplicates` | MUST | T-585 |  |
| F-586 | `test_data_cleaner.py` | Class `TestPreviewFunctions` | MUST | T-586 |  |
| F-587 | `test_data_cleaner.py` | Class `TestGetCleaningSummary` | MUST | T-587 |  |
| F-588 | `test_data_cleaner.py` | Class `TestEdgeCases` | MUST | T-588 |  |
| F-589 | `test_data_cleaner.py` | `sample_df`() | MUST | T-589 |  |
| F-590 | `test_data_cleaner.py` | `outlier_df`() | MUST | T-590 |  |
| F-591 | `test_data_cleaner.py` | `dup_df`() | MUST | T-591 |  |
| F-592 | `test_data_cleaner.py` | `empty_df`() | MUST | T-592 |  |
| F-593 | `test_data_cleaner.py` | `all_missing_df`() | MUST | T-593 |  |
| F-594 | `test_data_cleaner_extra.py` | Class `TestCleaningAction` | MUST | T-594 |  |
| F-595 | `test_data_cleaner_extra.py` | Class `TestDropColumns` | MUST | T-595 |  |
| F-596 | `test_data_cleaner_extra.py` | Class `TestDropRowsWithMissing` | MUST | T-596 |  |
| F-597 | `test_data_cleaner_extra.py` | Class `TestRemoveConstantColumns` | MUST | T-597 |  |
| F-598 | `test_data_cleaner_extra.py` | Class `TestClipOutliers` | MUST | T-598 |  |
| F-599 | `test_data_cleaner_extra.py` | Class `TestRemoveDuplicates` | MUST | T-599 |  |
| F-600 | `test_data_cleaner_extra.py` | Class `TestPreviewFunctions` | MUST | T-600 |  |
| F-601 | `test_descriptastorus_adapter.py` | Class `TestDescriptaStorusAdapter` | MUST | T-601 |  |
| F-602 | `test_dim_reduction.py` | Class `TestDimReducerPCA` | MUST | T-602 |  |
| F-603 | `test_dim_reduction.py` | Class `TestDimReducerTSNE` | MUST | T-603 |  |
| F-604 | `test_dim_reduction.py` | Class `TestDimReducerUMAP` | MUST | T-604 |  |
| F-605 | `test_dim_reduction.py` | `sample_df`() | MUST | T-605 |  |
| F-606 | `test_dim_reduction_extra.py` | Class `TestDimReductionConfig` | MUST | T-606 |  |
| F-607 | `test_dim_reduction_extra.py` | Class `TestDimReducerPCA` | MUST | T-607 |  |
| F-608 | `test_dim_reduction_extra.py` | Class `TestDimReducerTSNE` | MUST | T-608 |  |
| F-609 | `test_dim_reduction_extra.py` | Class `TestDimReducerUnknown` | MUST | T-609 |  |
| F-610 | `test_dim_reduction_extra.py` | Class `TestConvenienceFunctions` | MUST | T-610 |  |
| F-611 | `test_dim_reduction_final.py` | `test_pca_reconstruction_error`() | MUST | T-611 |  |
| F-612 | `test_dim_reduction_final.py` | `test_dim_reduction_extra_params`() | MUST | T-612 |  |
| F-613 | `test_domainml_analysis.py` | `test_metrics_satisfaction_score`() | MUST | T-613 |  |
| F-614 | `test_domainml_analysis.py` | `test_constrained_cv`() | MUST | T-614 |  |
| F-615 | `test_domainml_analysis.py` | `test_uncertainty_estimator`() | MUST | T-615 |  |
| F-616 | `test_domainml_engine.py` | `test_lazy_evaluator`() | MUST | T-616 |  |
| F-617 | `test_domainml_engine.py` | `test_monotonicity_engine`() | MUST | T-617 |  |
| F-618 | `test_domainml_kernel_opt.py` | `test_kernel_monotonicity_increasing`() | MUST | T-618 |  |
| F-619 | `test_domainml_kernel_opt.py` | `test_kernel_monotonicity_decreasing`() | MUST | T-619 |  |
| F-620 | `test_domainml_kernel_opt.py` | `test_kernel_invalid_inputs`() | MUST | T-620 |  |
| F-621 | `test_domainml_laplacian.py` | `test_sparse_laplacian_builder`() | MUST | T-621 |  |
| F-622 | `test_domainml_laplacian.py` | `test_manifold_validity_estimator`() | MUST | T-622 |  |
| F-623 | `test_e2e_smiles_to_ml.py` | Class `TestSmilesDescriptorTransform` | MUST | T-623 |  |
| F-624 | `test_e2e_smiles_to_ml.py` | Class `TestE2ERegressionSmilesToML` | MUST | T-624 |  |
| F-625 | `test_e2e_smiles_to_ml.py` | Class `TestE2EClassificationSmilesToML` | MUST | T-625 |  |
| F-626 | `test_e2e_smiles_to_ml.py` | Class `TestAutoTaskDetection` | MUST | T-626 |  |
| F-627 | `test_e2e_smiles_to_ml.py` | Class `TestAutoMLWithSmilesColDirect` | MUST | T-627 |  |
| F-628 | `test_e2e_smiles_to_ml.py` | Class `TestDescriptorNaNHandling` | MUST | T-628 |  |
| F-629 | `test_eda.py` | Class `TestComputeColumnStats` | MUST | T-629 |  |
| F-630 | `test_eda.py` | Class `TestSummarizeDataframe` | MUST | T-630 |  |
| F-631 | `test_eda.py` | Class `TestComputeCorrelation` | MUST | T-631 |  |
| F-632 | `test_eda.py` | Class `TestDetectOutliers` | MUST | T-632 |  |
| F-633 | `test_eda.py` | Class `TestComputeDistribution` | MUST | T-633 |  |
| F-634 | `test_eda.py` | Class `TestAnalyzeTarget` | MUST | T-634 |  |
| F-635 | `test_eda.py` | `mixed_df`() | MUST | T-635 |  |
| F-636 | `test_eda.py` | `outlier_df`() | MUST | T-636 |  |
| F-637 | `test_eda_extra.py` | Class `TestComputeColumnStats` | MUST | T-637 |  |
| F-638 | `test_eda_extra.py` | Class `TestSummarizeDataframe` | MUST | T-638 |  |
| F-639 | `test_eda_extra.py` | Class `TestComputeCorrelation` | MUST | T-639 |  |
| F-640 | `test_eda_extra.py` | Class `TestDetectOutliers` | MUST | T-640 |  |
| F-641 | `test_eda_extra.py` | Class `TestComputeDistribution` | MUST | T-641 |  |
| F-642 | `test_eda_extra.py` | Class `TestAnalyzeTarget` | MUST | T-642 |  |
| F-643 | `test_factory.py` | Class `TestGetModel` | MUST | T-643 |  |
| F-644 | `test_factory.py` | Class `TestListModels` | MUST | T-644 |  |
| F-645 | `test_factory.py` | Class `TestGetDefaultAutoMLModels` | MUST | T-645 |  |
| F-646 | `test_factory.py` | Class `TestGetModelRegistry` | MUST | T-646 |  |
| F-647 | `test_factory.py` | Class `TestOptionalModels` | MUST | T-647 |  |
| F-648 | `test_factory_comprehensive.py` | Class `TestGetModel` | MUST | T-648 |  |
| F-649 | `test_factory_comprehensive.py` | Class `TestListModels` | MUST | T-649 |  |
| F-650 | `test_factory_comprehensive.py` | Class `TestDefaultAutoml` | MUST | T-650 |  |
| F-651 | `test_factory_comprehensive.py` | Class `TestGetModelRegistry` | MUST | T-651 |  |
| F-652 | `test_factory_comprehensive.py` | Class `TestGetRegistry` | MUST | T-652 |  |
| F-653 | `test_factory_extra.py` | Class `TestGetRegistry` | MUST | T-653 |  |
| F-654 | `test_factory_extra.py` | Class `TestGetModel` | MUST | T-654 |  |
| F-655 | `test_factory_extra.py` | Class `TestListModels` | MUST | T-655 |  |
| F-656 | `test_factory_extra.py` | Class `TestDefaultAutoMLModels` | MUST | T-656 |  |
| F-657 | `test_feature_engineer.py` | Class `TestInteractionTransformer` | MUST | T-657 |  |
| F-658 | `test_feature_engineer.py` | Class `TestDatetimeFeatureExtractor` | MUST | T-658 |  |
| F-659 | `test_feature_engineer.py` | Class `TestLagRollingTransformer` | MUST | T-659 |  |
| F-660 | `test_feature_engineer.py` | Class `TestFeatureEngineeringConfig` | MUST | T-660 |  |
| F-661 | `test_feature_engineer.py` | Class `TestGroupAggTransformer` | MUST | T-661 |  |
| F-662 | `test_feature_engineer.py` | `group_df`() | MUST | T-662 |  |
| F-663 | `test_feature_engineer.py` | `small_numeric_df`() | MUST | T-663 |  |
| F-664 | `test_feature_engineer.py` | `datetime_series`() | MUST | T-664 |  |
| F-665 | `test_feature_engineer_ext.py` | Class `TestInteractionTransformer` | MUST | T-665 |  |
| F-666 | `test_feature_engineer_ext.py` | Class `TestGroupAggTransformer` | MUST | T-666 |  |
| F-667 | `test_feature_engineer_ext.py` | Class `TestDatetimeFeatureExtractor` | MUST | T-667 |  |
| F-668 | `test_feature_engineer_ext.py` | Class `TestLagRollingTransformer` | MUST | T-668 |  |
| F-669 | `test_feature_engineer_ext.py` | Class `TestBuildFeatureEngineeringPipeline` | MUST | T-669 |  |
| F-670 | `test_feature_engineer_extra.py` | Class `TestInteractionTransformer` | MUST | T-670 |  |
| F-671 | `test_feature_engineer_extra.py` | Class `TestGroupAggTransformer` | MUST | T-671 |  |
| F-672 | `test_feature_engineer_extra.py` | Class `TestDatetimeFeatureExtractor` | MUST | T-672 |  |
| F-673 | `test_feature_engineer_extra.py` | Class `TestLagRollingTransformer` | MUST | T-673 |  |
| F-674 | `test_feature_engineer_extra.py` | Class `TestFeatureEngineeringConfig` | MUST | T-674 |  |
| F-675 | `test_feature_generator_extra.py` | Class `TestFeatureGenConfig` | MUST | T-675 |  |
| F-676 | `test_feature_generator_extra.py` | Class `TestFeatureGeneratorNone` | MUST | T-676 |  |
| F-677 | `test_feature_generator_extra.py` | Class `TestFeatureGeneratorPoly` | MUST | T-677 |  |
| F-678 | `test_feature_generator_extra.py` | Class `TestFeatureGeneratorInteraction` | MUST | T-678 |  |
| F-679 | `test_feature_integrity.py` | Class `TestFeatureIntegrity` | MUST | T-679 |  |
| F-680 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorConfig` | MUST | T-680 |  |
| F-681 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorNone` | MUST | T-681 |  |
| F-682 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorLasso` | MUST | T-682 |  |
| F-683 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorRidge` | MUST | T-683 |  |
| F-684 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorRF` | MUST | T-684 |  |
| F-685 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorKBest` | MUST | T-685 |  |
| F-686 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorPercentile` | MUST | T-686 |  |
| F-687 | `test_feature_selector_comprehensive.py` | Class `TestFeatureSelectorUnknown` | MUST | T-687 |  |
| F-688 | `test_feature_selector_comprehensive.py` | `reg_data`() | MUST | T-688 |  |
| F-689 | `test_feature_selector_comprehensive.py` | `cls_data`() | MUST | T-689 |  |
| F-690 | `test_feature_selector_extra.py` | Class `TestFeatureSelectorNone` | MUST | T-690 |  |
| F-691 | `test_feature_selector_extra.py` | Class `TestLasso` | MUST | T-691 |  |
| F-692 | `test_feature_selector_extra.py` | Class `TestRidge` | MUST | T-692 |  |
| F-693 | `test_feature_selector_extra.py` | Class `TestRandomForest` | MUST | T-693 |  |
| F-694 | `test_feature_selector_extra.py` | Class `TestXGBoost` | MUST | T-694 |  |
| F-695 | `test_feature_selector_extra.py` | Class `TestSelectPercentile` | MUST | T-695 |  |
| F-696 | `test_feature_selector_extra.py` | Class `TestSelectKBest` | MUST | T-696 |  |
| F-697 | `test_feature_selector_extra.py` | Class `TestSelectFromModel` | MUST | T-697 |  |
| F-698 | `test_feature_selector_extra.py` | Class `TestUnknownMethod` | MUST | T-698 |  |
| F-699 | `test_feature_selector_extra.py` | Class `TestNdarrayInput` | MUST | T-699 |  |
| F-700 | `test_feature_selector_extra.py` | Class `TestMaxFeatures` | MUST | T-700 |  |
| F-701 | `test_feature_selector_extra.py` | Class `TestReliefF` | MUST | T-701 |  |
| F-702 | `test_feature_selector_extra.py` | Class `TestBoruta` | MUST | T-702 |  |
| F-703 | `test_feature_selector_extra.py` | Class `TestGenetic` | MUST | T-703 |  |
| F-704 | `test_feature_selector_extra.py` | Class `TestGroupLasso` | MUST | T-704 |  |
| F-705 | `test_final_push.py` | `test_loader_remaining_branches`(tmp_path) | MUST | T-705 |  |
| F-706 | `test_final_push.py` | `test_cv_manager_remaining_branches`() | MUST | T-706 |  |
| F-707 | `test_final_push.py` | `test_tuner_remaining_branches`() | MUST | T-707 |  |
| F-708 | `test_final_push.py` | `test_factory_list_models_detailed`() | MUST | T-708 |  |
| F-709 | `test_final_push.py` | `test_optional_import_safe_exhaustive`() | MUST | T-709 |  |
| F-710 | `test_forward_inverse_combinations.py` | Class `TestForwardModelCVCombinations` | MUST | T-710 |  |
| F-711 | `test_forward_inverse_combinations.py` | Class `TestForwardModelPreprocessCombinations` | MUST | T-711 |  |
| F-712 | `test_forward_inverse_combinations.py` | Class `TestInverseMethodTargetCombinations` | MUST | T-712 |  |
| F-713 | `test_forward_inverse_combinations.py` | Class `TestInverseMethodConstraintCombinations` | MUST | T-713 |  |
| F-714 | `test_forward_inverse_combinations.py` | Class `TestForwardToInverseE2E` | MUST | T-714 |  |
| F-715 | `test_forward_inverse_combinations.py` | Class `TestEdgeCasesDetailed` | MUST | T-715 |  |
| F-716 | `test_forward_inverse_combinations.py` | Class `TestHighDimensionalInverse` | MUST | T-716 |  |
| F-717 | `test_forward_inverse_combinations.py` | Class `TestBayesianAcquisitionCombinations` | MUST | T-717 |  |
| F-718 | `test_forward_inverse_combinations.py` | Class `TestGAParameterCombinations` | MUST | T-718 |  |
| F-719 | `test_forward_inverse_combinations.py` | Class `TestDirichletParameterCombinations` | MUST | T-719 |  |
| F-720 | `test_forward_inverse_combinations.py` | Class `TestConstraintCombinations` | MUST | T-720 |  |
| F-721 | `test_forward_inverse_combinations.py` | Class `TestScoreCalculation` | MUST | T-721 |  |
| F-722 | `test_forward_inverse_combinations.py` | Class `TestFullPipelineE2E` | MUST | T-722 |  |
| F-723 | `test_forward_inverse_combinations.py` | Class `TestCVManagerCombinations` | MUST | T-723 |  |
| F-724 | `test_forward_inverse_combinations.py` | Class `TestAutoMLAdvanced` | MUST | T-724 |  |
| F-725 | `test_forward_inverse_combinations.py` | Class `TestBOAdditionalCoverage` | MUST | T-725 |  |
| F-726 | `test_forward_inverse_combinations.py` | Class `TestFactoryComprehensive` | MUST | T-726 |  |
| F-727 | `test_group_contrib.py` | Class `TestGroupContribAdapter` | MUST | T-727 |  |
| F-728 | `test_group_contrib.py` | Class `TestJobackInternals` | MUST | T-728 |  |
| F-729 | `test_integration.py` | Class `TestDataPipeline` | MUST | T-729 |  |
| F-730 | `test_integration.py` | Class `TestEDAPipeline` | MUST | T-730 |  |
| F-731 | `test_integration.py` | Class `TestModelTrainingPipeline` | MUST | T-731 |  |
| F-732 | `test_integration.py` | `reg_df`() | MUST | T-732 |  |
| F-733 | `test_integration.py` | `cls_df`() | MUST | T-733 |  |
| F-734 | `test_interpret.py` | Class `TestSRIDecomposer` | MUST | T-734 |  |
| F-735 | `test_interpret.py` | Class `TestSelectFeaturesByIndependence` | MUST | T-735 |  |
| F-736 | `test_interpret.py` | Class `TestShapExplainer` | MUST | T-736 |  |
| F-737 | `test_interpret.py` | `dummy_shap_result`() | MUST | T-737 |  |
| F-738 | `test_interpret.py` | `multiclass_shap_result`() | MUST | T-738 |  |
| F-739 | `test_inverse_optimizer.py` | Class `TestRandomOptimizer` | MUST | T-739 |  |
| F-740 | `test_inverse_optimizer.py` | Class `TestGridOptimizer` | MUST | T-740 |  |
| F-741 | `test_inverse_optimizer.py` | Class `TestBayesianOptimizer` | MUST | T-741 |  |
| F-742 | `test_inverse_optimizer.py` | Class `TestGAOptimizer` | MUST | T-742 |  |
| F-743 | `test_inverse_optimizer.py` | Class `TestDirichletOptimizer` | MUST | T-743 |  |
| F-744 | `test_inverse_optimizer.py` | Class `TestEdgeCases` | MUST | T-744 |  |
| F-745 | `test_leakage_detector.py` | Class `TestHatMatrix` | MUST | T-745 |  |
| F-746 | `test_leakage_detector.py` | Class `TestRBFGram` | MUST | T-746 |  |
| F-747 | `test_leakage_detector.py` | Class `TestRFProximity` | MUST | T-747 |  |
| F-748 | `test_leakage_detector.py` | Class `TestGroupConsistencyScore` | MUST | T-748 |  |
| F-749 | `test_leakage_detector.py` | Class `TestEstimateGroups` | MUST | T-749 |  |
| F-750 | `test_leakage_detector.py` | Class `TestDetectLeakage` | MUST | T-750 |  |
| F-751 | `test_leakage_detector_extra.py` | Class `TestComputeHatMatrix` | MUST | T-751 |  |
| F-752 | `test_leakage_detector_extra.py` | Class `TestComputeRBFGram` | MUST | T-752 |  |
| F-753 | `test_leakage_detector_extra.py` | Class `TestComputeRFProximity` | MUST | T-753 |  |
| F-754 | `test_leakage_detector_extra.py` | Class `TestFindSuspiciousPairs` | MUST | T-754 |  |
| F-755 | `test_leakage_detector_extra.py` | Class `TestGroupConsistency` | MUST | T-755 |  |
| F-756 | `test_leakage_detector_extra.py` | Class `TestEstimateGroups` | MUST | T-756 |  |
| F-757 | `test_leakage_detector_extra.py` | Class `TestDetectLeakage` | MUST | T-757 |  |
| F-758 | `test_leakage_detector_extra.py` | Class `TestCheckFeatureLeakage` | MUST | T-758 |  |
| F-759 | `test_linear_tree_extra.py` | Class `TestUtilities` | MUST | T-759 |  |
| F-760 | `test_linear_tree_extra.py` | Class `TestNode` | MUST | T-760 |  |
| F-761 | `test_linear_tree_extra.py` | Class `TestLinearTreeRegressor` | MUST | T-761 |  |
| F-762 | `test_linear_tree_extra.py` | Class `TestLinearTreeClassifier` | MUST | T-762 |  |
| F-763 | `test_linear_tree_extra.py` | Class `TestLinearForestRegressor` | MUST | T-763 |  |
| F-764 | `test_linear_tree_extra.py` | Class `TestLinearForestClassifier` | MUST | T-764 |  |
| F-765 | `test_linear_tree_extra.py` | Class `TestLinearBoostRegressor` | MUST | T-765 |  |
| F-766 | `test_linear_tree_extra.py` | Class `TestLinearBoostClassifier` | MUST | T-766 |  |
| F-767 | `test_linear_tree_rgf_monotonic.py` | Class `TestLinearTreeRegressor` | MUST | T-767 |  |
| F-768 | `test_linear_tree_rgf_monotonic.py` | Class `TestLinearTreeClassifier` | MUST | T-768 |  |
| F-769 | `test_linear_tree_rgf_monotonic.py` | Class `TestLinearForestRegressor` | MUST | T-769 |  |
| F-770 | `test_linear_tree_rgf_monotonic.py` | Class `TestLinearBoostRegressor` | MUST | T-770 |  |
| F-771 | `test_linear_tree_rgf_monotonic.py` | Class `TestLinearBoostClassifier` | MUST | T-771 |  |
| F-772 | `test_linear_tree_rgf_monotonic.py` | Class `TestRGFRegressor` | MUST | T-772 |  |
| F-773 | `test_linear_tree_rgf_monotonic.py` | Class `TestRGFClassifier` | MUST | T-773 |  |
| F-774 | `test_linear_tree_rgf_monotonic.py` | Class `TestMonotonicKernelWrapper` | MUST | T-774 |  |
| F-775 | `test_linear_tree_rgf_monotonic.py` | Class `TestPipelineIntegration` | MUST | T-775 |  |
| F-776 | `test_loader_extra.py` | Class `TestLoadFile` | MUST | T-776 |  |
| F-777 | `test_loader_extra.py` | Class `TestLoadFromBytes` | MUST | T-777 |  |
| F-778 | `test_loader_extra.py` | Class `TestSaveDataframe` | MUST | T-778 |  |
| F-779 | `test_loader_extra.py` | Class `TestSupportedExtensions` | MUST | T-779 |  |
| F-780 | `test_mlops.py` | `mock_mlflow`() | MUST | T-780 |  |
| F-781 | `test_mlops.py` | `mlflow_manager`() | MUST | T-781 |  |
| F-782 | `test_mlops.py` | `test_mlflow_manager_init`(mock_mlflow, mlflow_manager) | MUST | T-782 |  |
| F-783 | `test_mlops.py` | `test_start_end_run`(mock_mlflow, mlflow_manager) | MUST | T-783 |  |
| F-784 | `test_mlops.py` | `test_log_params_metrics`(mock_mlflow, mlflow_manager) | MUST | T-784 |  |
| F-785 | `test_mlops.py` | `test_mlrun_context`(mock_mlflow, mlflow_manager) | MUST | T-785 |  |
| F-786 | `test_mlops.py` | `test_run_context_fail`(mock_mlflow, mlflow_manager) | MUST | T-786 |  |
| F-787 | `test_mlops.py` | `test_save_and_load_model`(mock_mlflow, mlflow_manager, tmp_path) | MUST | T-787 |  |
| F-788 | `test_mlops.py` | `test_get_experiment_runs`(mock_mlflow, mlflow_manager) | MUST | T-788 |  |
| F-789 | `test_mlops.py` | `test_log_figure`(mock_mlflow, mlflow_manager) | MUST | T-789 |  |
| F-790 | `test_mlops.py` | `test_best_run_none`(mock_mlflow, mlflow_manager) | MUST | T-790 |  |
| F-791 | `test_mlops.py` | `test_save_model_registry_fail`(mock_mlflow, mlflow_manager, tmp_path) | MUST | T-791 |  |
| F-792 | `test_mlops.py` | `test_log_artifact_no_mlflow`(tmp_path) | MUST | T-792 |  |
| F-793 | `test_mlops.py` | `test_run_context_with_tags`(mock_mlflow, mlflow_manager) | MUST | T-793 |  |
| F-794 | `test_mlops_extended.py` | Class `TestMLflowManagerGracefulDegradation` | MUST | T-794 |  |
| F-795 | `test_mlops_extended.py` | Class `TestModelSaveLoad` | MUST | T-795 |  |
| F-796 | `test_mlops_extended.py` | Class `TestMLRunContext` | MUST | T-796 |  |
| F-797 | `test_mlops_extended.py` | Class `TestCosmoAdapterMock` | MUST | T-797 |  |
| F-798 | `test_models.py` | Class `TestModelFactory` | MUST | T-798 |  |
| F-799 | `test_models.py` | Class `TestCVManager` | MUST | T-799 |  |
| F-800 | `test_models.py` | Class `TestTuner` | MUST | T-800 |  |
| F-801 | `test_models.py` | Class `TestAutoMLEngine` | MUST | T-801 |  |
| F-802 | `test_models.py` | `regression_data`() | MUST | T-802 |  |
| F-803 | `test_models.py` | `classification_data`() | MUST | T-803 |  |
| F-804 | `test_models.py` | `regression_df`() | MUST | T-804 |  |
| F-805 | `test_models.py` | `generate_numeric_data`(n_samples, n_features, n_classes, task) | MUST | T-805 |  |
| F-806 | `test_mol2vec_adapter.py` | Class `TestMol2VecAdapter` | MUST | T-806 |  |
| F-807 | `test_molfeat_adapter.py` | Class `TestMolfeatAdapter` | MUST | T-807 |  |
| F-808 | `test_monotonic_kernel.py` | Class `TestComputeMonotonicViolation` | MUST | T-808 |  |
| F-809 | `test_monotonic_kernel.py` | Class `TestMonotonicKernelWrapper` | MUST | T-809 |  |
| F-810 | `test_monotonic_kernel.py` | Class `TestMonotonicKernelClassifierWrapper` | MUST | T-810 |  |
| F-811 | `test_monotonic_kernel.py` | Class `TestWrapFactory` | MUST | T-811 |  |
| F-812 | `test_monotonic_kernel.py` | Class `TestBuildGridX` | MUST | T-812 |  |
| F-813 | `test_monotonic_kernel.py` | `monotonic_regression_data`() | MUST | T-813 |  |
| F-814 | `test_monotonic_kernel.py` | `classification_data`() | MUST | T-814 |  |
| F-815 | `test_monotonic_kernel_comprehensive.py` | Class `TestComputeMonotonicViolation` | MUST | T-815 |  |
| F-816 | `test_monotonic_kernel_comprehensive.py` | Class `TestBuildGridX` | MUST | T-816 |  |
| F-817 | `test_monotonic_kernel_comprehensive.py` | Class `TestFitWithWeight` | MUST | T-817 |  |
| F-818 | `test_monotonic_kernel_comprehensive.py` | Class `TestIsSoftMonotonicCandidate` | MUST | T-818 |  |
| F-819 | `test_monotonic_kernel_comprehensive.py` | Class `TestMonotonicKernelWrapper` | MUST | T-819 |  |
| F-820 | `test_monotonic_kernel_comprehensive.py` | Class `TestMonotonicKernelClassifierWrapper` | MUST | T-820 |  |
| F-821 | `test_monotonic_kernel_comprehensive.py` | Class `TestWrapWithSoftMonotonic` | MUST | T-821 |  |
| F-822 | `test_monotonic_kernel_extra.py` | Class `TestToNumpy` | MUST | T-822 |  |
| F-823 | `test_monotonic_kernel_extra.py` | Class `TestComputeViolation` | MUST | T-823 |  |
| F-824 | `test_monotonic_kernel_extra.py` | Class `TestBuildGridX` | MUST | T-824 |  |
| F-825 | `test_monotonic_kernel_extra.py` | Class `TestBuildMonotonicAugmentedData` | MUST | T-825 |  |
| F-826 | `test_monotonic_kernel_extra.py` | Class `TestMonotonicKernelWrapper` | MUST | T-826 |  |
| F-827 | `test_monotonic_kernel_extra.py` | Class `TestMonotonicKernelClassifierWrapper` | MUST | T-827 |  |
| F-828 | `test_monotonic_kernel_extra.py` | Class `TestFitWithWeight` | MUST | T-828 |  |
| F-829 | `test_monotonic_kernel_extra.py` | Class `TestSoftMonotonicCandidate` | MUST | T-829 |  |
| F-830 | `test_monotonic_kernel_extra.py` | Class `TestWrapWithSoftMonotonic` | MUST | T-830 |  |
| F-831 | `test_monotonic_wrapper.py` | Class `TestMonotonicConstraintRegressorAPI` | MUST | T-831 |  |
| F-832 | `test_monotonic_wrapper.py` | Class `TestMonotonicConstraintRegressorMonotonicity` | MUST | T-832 |  |
| F-833 | `test_monotonic_wrapper.py` | Class `TestMonotonicConstraintClassifierAPI` | MUST | T-833 |  |
| F-834 | `test_monotonic_wrapper.py` | Class `TestModelMonotonicStrategy` | MUST | T-834 |  |
| F-835 | `test_monotonic_wrapper.py` | Class `TestWrapMonotonic` | MUST | T-835 |  |
| F-836 | `test_monotonic_wrapper.py` | Class `TestApplyMonotonicConstraintsRouting` | MUST | T-836 |  |
| F-837 | `test_monotonic_wrapper.py` | Class `TestColumnMetaIntegrationMonotonic` | MUST | T-837 |  |
| F-838 | `test_monotonic_wrapper.py` | Class `TestAutoDetectMonotonic` | MUST | T-838 |  |
| F-839 | `test_monotonic_wrapper.py` | Class `TestConstraintStrength` | MUST | T-839 |  |
| F-840 | `test_monotonic_wrapper.py` | Class `TestExtrapolationMonotonicity` | MUST | T-840 |  |
| F-841 | `test_monotonic_wrapper.py` | `regression_data`() | MUST | T-841 |  |
| F-842 | `test_monotonic_wrapper.py` | `classification_data`() | MUST | T-842 |  |
| F-843 | `test_mordred_adapter.py` | Class `TestMordredAdapterProperties` | MUST | T-843 |  |
| F-844 | `test_mordred_adapter.py` | Class `TestDescriptorNames` | MUST | T-844 |  |
| F-845 | `test_mordred_adapter.py` | Class `TestMordredCompute` | MUST | T-845 |  |
| F-846 | `test_nicegui_components.py` | Class `TestAutoDetectColumns` | MUST | T-846 |  |
| F-847 | `test_nicegui_components.py` | Class `TestToggleModel` | MUST | T-847 |  |
| F-848 | `test_nicegui_components.py` | Class `TestOnTargetChange` | MUST | T-848 |  |
| F-849 | `test_nicegui_components.py` | Class `TestSampleSmiles` | MUST | T-849 |  |
| F-850 | `test_nicegui_components.py` | Class `TestRunEngineSync` | MUST | T-850 |  |
| F-851 | `test_nicegui_components.py` | Class `TestAllEngines` | MUST | T-851 |  |
| F-852 | `test_nicegui_components.py` | Class `TestEngineInfo` | MUST | T-852 |  |
| F-853 | `test_nicegui_components.py` | Class `TestAutoParamsUiImport` | MUST | T-853 |  |
| F-854 | `test_optim_comprehensive.py` | Class `TestRangeConstraint` | MUST | T-854 |  |
| F-855 | `test_optim_comprehensive.py` | Class `TestSumConstraint` | MUST | T-855 |  |
| F-856 | `test_optim_comprehensive.py` | Class `TestInequalityConstraint` | MUST | T-856 |  |
| F-857 | `test_optim_comprehensive.py` | Class `TestAtLeastNConstraint` | MUST | T-857 |  |
| F-858 | `test_optim_comprehensive.py` | Class `TestCustomConstraint` | MUST | T-858 |  |
| F-859 | `test_optim_comprehensive.py` | Class `TestApplyConstraints` | MUST | T-859 |  |
| F-860 | `test_optim_comprehensive.py` | Class `TestVariable` | MUST | T-860 |  |
| F-861 | `test_optim_comprehensive.py` | Class `TestSearchSpace` | MUST | T-861 |  |
| F-862 | `test_optional_import_extra.py` | Class `TestSafeImport` | MUST | T-862 |  |
| F-863 | `test_optional_import_extra.py` | Class `TestIsAvailable` | MUST | T-863 |  |
| F-864 | `test_optional_import_extra.py` | Class `TestRequire` | MUST | T-864 |  |
| F-865 | `test_optional_import_extra.py` | Class `TestGetReport` | MUST | T-865 |  |
| F-866 | `test_optional_import_extra.py` | Class `TestProbeAll` | MUST | T-866 |  |
| F-867 | `test_padel_adapter.py` | Class `TestPaDELAdapter` | MUST | T-867 |  |
| F-868 | `test_param_schema.py` | `test_sklearn_estimators`() | MUST | T-868 |  |
| F-869 | `test_param_schema.py` | `test_chem_adapters`() | MUST | T-869 |  |
| F-870 | `test_param_schema.py` | `test_feature_selection_preprocessing`() | MUST | T-870 |  |
| F-871 | `test_param_schema.py` | `test_apply_params`() | MUST | T-871 |  |
| F-872 | `test_param_schema.py` | `test_to_dict_json`() | MUST | T-872 |  |
| F-873 | `test_param_schema.py` | `test_new_model_auto_detection`() | MUST | T-873 |  |
| F-874 | `test_param_schema.py` | `test_factory_integration`() | MUST | T-874 |  |
| F-875 | `test_param_schema_comprehensive.py` | Class `TestParamSpec` | MUST | T-875 |  |
| F-876 | `test_param_schema_comprehensive.py` | Class `TestIntrospectParams` | MUST | T-876 |  |
| F-877 | `test_param_schema_comprehensive.py` | Class `TestApplyParams` | MUST | T-877 |  |
| F-878 | `test_param_schema_comprehensive.py` | Class `TestConvertValue` | MUST | T-878 |  |
| F-879 | `test_param_schema_comprehensive.py` | Class `TestGroupFilters` | MUST | T-879 |  |
| F-880 | `test_param_schema_comprehensive.py` | Class `TestDocstringExtraction` | MUST | T-880 |  |
| F-881 | `test_param_schema_extra.py` | Class `SimpleModel` | MUST | T-881 |  |
| F-882 | `test_param_schema_extra.py` | Class `ModelWithOptionalHints` | MUST | T-882 |  |
| F-883 | `test_param_schema_extra.py` | Class `EnumModel` | MUST | T-883 |  |
| F-884 | `test_param_schema_extra.py` | Class `LiteralModel` | MUST | T-884 |  |
| F-885 | `test_param_schema_extra.py` | Class `ListParamModel` | MUST | T-885 |  |
| F-886 | `test_param_schema_extra.py` | Class `DictParamModel` | MUST | T-886 |  |
| F-887 | `test_param_schema_extra.py` | Class `NumpydocModel` | MUST | T-887 |  |
| F-888 | `test_param_schema_extra.py` | Class `RstDocModel` | MUST | T-888 |  |
| F-889 | `test_param_schema_extra.py` | Class `NoDocModel` | MUST | T-889 |  |
| F-890 | `test_param_schema_extra.py` | Class `VarArgModel` | MUST | T-890 |  |
| F-891 | `test_param_schema_extra.py` | Class `PrivateParamModel` | MUST | T-891 |  |
| F-892 | `test_param_schema_extra.py` | Class `TestIntrospectParams` | MUST | T-892 |  |
| F-893 | `test_param_schema_extra.py` | Class `TestIntrospectAdapter` | MUST | T-893 |  |
| F-894 | `test_param_schema_extra.py` | Class `TestDocstringParsing` | MUST | T-894 |  |
| F-895 | `test_param_schema_extra.py` | Class `TestParamSpecToDict` | MUST | T-895 |  |
| F-896 | `test_param_schema_extra.py` | Class `TestApplyParams` | MUST | T-896 |  |
| F-897 | `test_param_schema_extra.py` | Class `TestConvertValue` | MUST | T-897 |  |
| F-898 | `test_param_schema_extra.py` | Class `TestGroupFiltering` | MUST | T-898 |  |
| F-899 | `test_pipeline.py` | Class `TestColumnSelectorWrapper` | MUST | T-899 |  |
| F-900 | `test_pipeline.py` | Class `TestColPreprocessor` | MUST | T-900 |  |
| F-901 | `test_pipeline.py` | Class `TestFeatureGenerator` | MUST | T-901 |  |
| F-902 | `test_pipeline.py` | Class `TestFeatureSelector` | MUST | T-902 |  |
| F-903 | `test_pipeline.py` | Class `TestBuildPipeline` | MUST | T-903 |  |
| F-904 | `test_pipeline.py` | Class `TestMonotonicAndGroups` | MUST | T-904 |  |
| F-905 | `test_pipeline.py` | `simple_numeric_df`() | MUST | T-905 |  |
| F-906 | `test_pipeline.py` | `mixed_df`() | MUST | T-906 |  |
| F-907 | `test_pipeline.py` | `regression_data`() | MUST | T-907 |  |
| F-908 | `test_pipeline.py` | `classification_data`() | MUST | T-908 |  |
| F-909 | `test_pipeline_builder_extra.py` | Class `TestPipelineConfig` | MUST | T-909 |  |
| F-910 | `test_pipeline_builder_extra.py` | Class `TestBuildPipeline` | MUST | T-910 |  |
| F-911 | `test_pipeline_builder_extra.py` | Class `TestApplyMonotonicConstraints` | MUST | T-911 |  |
| F-912 | `test_pipeline_builder_extra.py` | Class `TestExtractGroupArray` | MUST | T-912 |  |
| F-913 | `test_pipeline_comprehensive.py` | Class `TestColumnMeta` | MUST | T-913 |  |
| F-914 | `test_pipeline_comprehensive.py` | Class `TestColumnSelectorWrapper` | MUST | T-914 |  |
| F-915 | `test_pipeline_comprehensive.py` | Class `TestExtractGroupArray` | MUST | T-915 |  |
| F-916 | `test_pipeline_comprehensive.py` | Class `TestPipelineConfig` | MUST | T-916 |  |
| F-917 | `test_pipeline_comprehensive.py` | Class `TestBuildPipeline` | MUST | T-917 |  |
| F-918 | `test_pipeline_comprehensive.py` | Class `TestApplyMonotonicConstraints` | MUST | T-918 |  |
| F-919 | `test_pipeline_grid.py` | Class `TestCountCombinations` | MUST | T-919 |  |
| F-920 | `test_pipeline_grid.py` | Class `TestSingleCombination` | MUST | T-920 |  |
| F-921 | `test_pipeline_grid.py` | Class `TestMultipleImputerEstimator` | MUST | T-921 |  |
| F-922 | `test_pipeline_grid.py` | Class `TestFullGridAxes` | MUST | T-922 |  |
| F-923 | `test_pipeline_grid.py` | Class `TestFeatureGenGrid` | MUST | T-923 |  |
| F-924 | `test_pipeline_grid.py` | Class `TestFeatureSelGrid` | MUST | T-924 |  |
| F-925 | `test_pipeline_grid.py` | Class `TestMaxCombinations` | MUST | T-925 |  |
| F-926 | `test_pipeline_grid.py` | Class `TestEndToEnd` | MUST | T-926 |  |
| F-927 | `test_pipeline_grid.py` | Class `TestEstimatorParamsList` | MUST | T-927 |  |
| F-928 | `test_pipeline_grid.py` | Class `TestClassificationGrid` | MUST | T-928 |  |
| F-929 | `test_pipeline_grid.py` | Class `TestCatImputersGrid` | MUST | T-929 |  |
| F-930 | `test_pipeline_grid.py` | Class `TestBinaryImputersGrid` | MUST | T-930 |  |
| F-931 | `test_pipeline_grid.py` | Class `TestExcludeColumns` | MUST | T-931 |  |
| F-932 | `test_pipeline_grid.py` | `reg_df`() | MUST | T-932 |  |
| F-933 | `test_pipeline_grid.py` | `clf_df`() | MUST | T-933 |  |
| F-934 | `test_pipeline_grid_comprehensive.py` | Class `TestPipelineGridConfig` | MUST | T-934 |  |
| F-935 | `test_pipeline_grid_comprehensive.py` | Class `TestCountCombinations` | MUST | T-935 |  |
| F-936 | `test_pipeline_grid_comprehensive.py` | Class `TestEnsureNonempty` | MUST | T-936 |  |
| F-937 | `test_pipeline_grid_comprehensive.py` | Class `TestMakeLabel` | MUST | T-937 |  |
| F-938 | `test_pipeline_grid_comprehensive.py` | Class `TestBuildGenCombinations` | MUST | T-938 |  |
| F-939 | `test_pipeline_grid_comprehensive.py` | Class `TestGeneratePipelineGrid` | MUST | T-939 |  |
| F-940 | `test_pipeline_grid_extra.py` | Class `TestPipelineGridConfig` | MUST | T-940 |  |
| F-941 | `test_pipeline_grid_extra.py` | Class `TestHelpers` | MUST | T-941 |  |
| F-942 | `test_pipeline_grid_extra.py` | Class `TestCountCombinations` | MUST | T-942 |  |
| F-943 | `test_pipeline_grid_extra.py` | Class `TestGeneratePipelineGrid` | MUST | T-943 |  |
| F-944 | `test_pipeline_modules.py` | Class `TestColPreprocessor` | MUST | T-944 |  |
| F-945 | `test_pipeline_modules.py` | Class `TestColumnSelectorWrapper` | MUST | T-945 |  |
| F-946 | `test_pipeline_modules.py` | Class `TestFeatureGenerator` | MUST | T-946 |  |
| F-947 | `test_pipeline_modules.py` | Class `TestFeatureSelector` | MUST | T-947 |  |
| F-948 | `test_pipeline_modules.py` | `mixed_df`() | MUST | T-948 |  |
| F-949 | `test_pipeline_modules.py` | `numeric_df`() | MUST | T-949 |  |
| F-950 | `test_pipeline_modules.py` | `regression_target`(numeric_df) | MUST | T-950 |  |
| F-951 | `test_preprocessor_comprehensive.py` | Class `TestLogTransformer` | MUST | T-951 |  |
| F-952 | `test_preprocessor_comprehensive.py` | Class `TestSinCosTransformer` | MUST | T-952 |  |
| F-953 | `test_preprocessor_comprehensive.py` | Class `TestPreprocessConfig` | MUST | T-953 |  |
| F-954 | `test_preprocessor_comprehensive.py` | Class `TestPreprocessor` | MUST | T-954 |  |
| F-955 | `test_preprocessor_comprehensive.py` | Class `TestBuildFullPipeline` | MUST | T-955 |  |
| F-956 | `test_preprocessor_comprehensive.py` | `mixed_df`() | MUST | T-956 |  |
| F-957 | `test_preprocessor_comprehensive.py` | `detection_result`(mixed_df) | MUST | T-957 |  |
| F-958 | `test_preprocessor_extra.py` | Class `TestLogTransformer` | MUST | T-958 |  |
| F-959 | `test_preprocessor_extra.py` | Class `TestSinCosTransformer` | MUST | T-959 |  |
| F-960 | `test_preprocessor_extra.py` | Class `TestPreprocessConfig` | MUST | T-960 |  |
| F-961 | `test_preprocessor_extra.py` | Class `TestPreprocessor` | MUST | T-961 |  |
| F-962 | `test_preprocessor_extra.py` | Class `TestBuildFullPipeline` | MUST | T-962 |  |
| F-963 | `test_preset_manager.py` | Class `TestSavePreset` | MUST | T-963 |  |
| F-964 | `test_preset_manager.py` | Class `TestLoadPreset` | MUST | T-964 |  |
| F-965 | `test_preset_manager.py` | Class `TestListPresets` | MUST | T-965 |  |
| F-966 | `test_preset_manager.py` | Class `TestDeletePreset` | MUST | T-966 |  |
| F-967 | `test_preset_manager.py` | Class `TestExportStateSummary` | MUST | T-967 |  |
| F-968 | `test_preset_manager.py` | Class `TestMakeSerializable` | MUST | T-968 |  |
| F-969 | `test_preset_manager.py` | Class `TestRecordAnalysis` | MUST | T-969 |  |
| F-970 | `test_preset_manager.py` | Class `TestListHistory` | MUST | T-970 |  |
| F-971 | `test_preset_manager.py` | Class `TestExportImportConfigYaml` | MUST | T-971 |  |
| F-972 | `test_preset_manager.py` | `tmp_preset_dir`(tmp_path) | MUST | T-972 |  |
| F-973 | `test_preset_manager.py` | `sample_state`() | MUST | T-973 |  |
| F-974 | `test_protonation.py` | Class `_FakeConfig` | MUST | T-974 |  |
| F-975 | `test_protonation.py` | Class `TestNeutralize` | MUST | T-975 |  |
| F-976 | `test_protonation.py` | Class `TestMaxDeprotonate` | MUST | T-976 |  |
| F-977 | `test_protonation.py` | Class `TestMaxProtonate` | MUST | T-977 |  |
| F-978 | `test_protonation.py` | Class `TestApplyProtonation` | MUST | T-978 |  |
| F-979 | `test_protonation.py` | Class `TestApplyProtonationBatch` | MUST | T-979 |  |
| F-980 | `test_psmiles_adapter.py` | Class `TestPSmilesAdapter` | MUST | T-980 |  |
| F-981 | `test_random_projection.py` | Class `TestJLRandomProjectionPassthrough` | MUST | T-981 |  |
| F-982 | `test_random_projection.py` | Class `TestJLRandomProjectionActive` | MUST | T-982 |  |
| F-983 | `test_random_projection.py` | Class `TestSummary` | MUST | T-983 |  |
| F-984 | `test_random_projection.py` | Class `TestFeatureNamesOut` | MUST | T-984 |  |
| F-985 | `test_random_projection.py` | Class `TestMethodAuto` | MUST | T-985 |  |
| F-986 | `test_random_projection.py` | Class `TestMethodFixed` | MUST | T-986 |  |
| F-987 | `test_random_projection.py` | Class `TestShouldApply` | MUST | T-987 |  |
| F-988 | `test_random_projection.py` | Class `TestBuildPipelineWithJLRP` | MUST | T-988 |  |
| F-989 | `test_random_projection.py` | Class `TestBuildPipelineWithoutJLRP` | MUST | T-989 |  |
| F-990 | `test_random_projection.py` | Class `TestPreprocessConfigRP` | MUST | T-990 |  |
| F-991 | `test_random_projection.py` | Class `TestRunMultiFeatureSets` | MUST | T-991 |  |
| F-992 | `test_random_projection.py` | Class `TestNotFittedError` | MUST | T-992 |  |
| F-993 | `test_random_projection.py` | Class `TestSparseInput` | MUST | T-993 |  |
| F-994 | `test_random_projection.py` | `low_dim_X`() | MUST | T-994 |  |
| F-995 | `test_random_projection.py` | `high_dim_X`() | MUST | T-995 |  |
| F-996 | `test_random_projection.py` | `regression_df`() | MUST | T-996 |  |
| F-997 | `test_recommender.py` | Class `TestRecommenderDatabase` | MUST | T-997 |  |
| F-998 | `test_recommender.py` | Class `TestGetTargetRecommendationByName` | MUST | T-998 |  |
| F-999 | `test_recommender.py` | Class `TestGetTargetNames` | MUST | T-999 |  |
| F-1000 | `test_recommender.py` | Class `TestGetTargetCategories` | MUST | T-1000 |  |
| F-1001 | `test_recommender.py` | Class `TestGetTargetsByCategory` | MUST | T-1001 |  |
| F-1002 | `test_recommender.py` | Class `TestGetAllDescriptorCategories` | MUST | T-1002 |  |
| F-1003 | `test_recommender.py` | Class `TestGetDescriptorsByCategory` | MUST | T-1003 |  |
| F-1004 | `test_recommender.py` | Class `TestSpecificTargets` | MUST | T-1004 |  |
| F-1005 | `test_recommender_comprehensive.py` | Class `TestDescriptorInfo` | MUST | T-1005 |  |
| F-1006 | `test_recommender_comprehensive.py` | Class `TestTargetRecommendations` | MUST | T-1006 |  |
| F-1007 | `test_recommender_comprehensive.py` | Class `TestGetAllRecommendations` | MUST | T-1007 |  |
| F-1008 | `test_recommender_comprehensive.py` | Class `TestGetByName` | MUST | T-1008 |  |
| F-1009 | `test_recommender_comprehensive.py` | Class `TestGetTargetNames` | MUST | T-1009 |  |
| F-1010 | `test_recommender_comprehensive.py` | Class `TestGetTargetCategories` | MUST | T-1010 |  |
| F-1011 | `test_recommender_comprehensive.py` | Class `TestGetTargetsByCategory` | MUST | T-1011 |  |
| F-1012 | `test_recommender_comprehensive.py` | Class `TestGetDescriptorCategories` | MUST | T-1012 |  |
| F-1013 | `test_recommender_comprehensive.py` | Class `TestGetDescriptorsByCategory` | MUST | T-1013 |  |
| F-1014 | `test_rgf_comprehensive.py` | Class `TestUtilities` | MUST | T-1014 |  |
| F-1015 | `test_rgf_comprehensive.py` | Class `TestRGFRegressor` | MUST | T-1015 |  |
| F-1016 | `test_rgf_comprehensive.py` | Class `TestRGFClassifier` | MUST | T-1016 |  |
| F-1017 | `test_rgf_comprehensive.py` | Class `TestRGFCore` | MUST | T-1017 |  |
| F-1018 | `test_rgf_extra.py` | Class `TestRGFUtilities` | MUST | T-1018 |  |
| F-1019 | `test_rgf_extra.py` | Class `TestRGFRegressor` | MUST | T-1019 |  |
| F-1020 | `test_rgf_extra.py` | Class `TestRGFClassifier` | MUST | T-1020 |  |
| F-1021 | `test_search_space_extra.py` | Class `TestVariable` | MUST | T-1021 |  |
| F-1022 | `test_search_space_extra.py` | Class `TestSearchSpaceBasic` | MUST | T-1022 |  |
| F-1023 | `test_search_space_extra.py` | Class `TestGenerateCandidates` | MUST | T-1023 |  |
| F-1024 | `test_search_space_extra.py` | Class `TestFromDataFrame` | MUST | T-1024 |  |
| F-1025 | `test_search_space_generator.py` | Class `TestGenerateGridSpace` | MUST | T-1025 |  |
| F-1026 | `test_search_space_generator.py` | Class `TestGenerateOptunaSpace` | MUST | T-1026 |  |
| F-1027 | `test_search_space_generator.py` | Class `TestGenerateSearchSpaces` | MUST | T-1027 |  |
| F-1028 | `test_search_space_generator.py` | Class `TestFromEstimator` | MUST | T-1028 |  |
| F-1029 | `test_search_space_generator.py` | Class `TestKnownPresets` | MUST | T-1029 |  |
| F-1030 | `test_search_space_generator.py` | Class `TestAutoInference` | MUST | T-1030 |  |
| F-1031 | `test_search_space_generator.py` | Class `TestSearchParamSpecMethods` | MUST | T-1031 |  |
| F-1032 | `test_search_space_generator.py` | Class `TestParseValueList` | MUST | T-1032 |  |
| F-1033 | `test_search_space_generator.py` | Class `TestRangeInference` | MUST | T-1033 |  |
| F-1034 | `test_search_space_generator.py` | Class `TestEstimatorConfig` | MUST | T-1034 |  |
| F-1035 | `test_search_space_generator.py` | Class `TestDocstringIntegration` | MUST | T-1035 |  |
| F-1036 | `test_search_space_generator.py` | Class `TestEndToEnd` | MUST | T-1036 |  |
| F-1037 | `test_shap_explainer_extra.py` | Class `TestShapConfig` | MUST | T-1037 |  |
| F-1038 | `test_shap_explainer_extra.py` | Class `TestShapResult` | MUST | T-1038 |  |
| F-1039 | `test_shap_explainer_extra.py` | Class `TestShapExplainer` | MUST | T-1039 |  |
| F-1040 | `test_skfp_adapter.py` | Class `TestSkfpAdapter` | MUST | T-1040 |  |
| F-1041 | `test_smiles_pipeline.py` | Class `TestSmilesAutoMLPipeline` | MUST | T-1041 |  |
| F-1042 | `test_smiles_pipeline.py` | `test_smiles_pipeline_pre_expanded_fit`() | MUST | T-1042 |  |
| F-1043 | `test_smiles_transformer.py` | Class `TestSmilesDescriptorTransformer` | MUST | T-1043 |  |
| F-1044 | `test_smiles_transformer.py` | Class `TestPrecalculateFunctions` | MUST | T-1044 |  |
| F-1045 | `test_smiles_transformer.py` | `smiles_df`() | MUST | T-1045 |  |
| F-1046 | `test_sri.py` | Class `_MockShapResult` | MUST | T-1046 |  |
| F-1047 | `test_sri.py` | Class `TestSRIDecomposer` | MUST | T-1047 |  |
| F-1048 | `test_sri.py` | Class `TestSRIResult` | MUST | T-1048 |  |
| F-1049 | `test_sri.py` | Class `TestSelectFeatures` | MUST | T-1049 |  |
| F-1050 | `test_sri_extra.py` | Class `TestSRIDecomposer` | MUST | T-1050 |  |
| F-1051 | `test_sri_extra.py` | Class `TestSRIResult` | MUST | T-1051 |  |
| F-1052 | `test_sri_extra.py` | Class `TestSelectFeatures` | MUST | T-1052 |  |
| F-1053 | `test_tuner.py` | Class `TestTunerConfig` | MUST | T-1053 |  |
| F-1054 | `test_tuner.py` | Class `TestGridSearch` | MUST | T-1054 |  |
| F-1055 | `test_tuner.py` | Class `TestRandomSearch` | MUST | T-1055 |  |
| F-1056 | `test_tuner.py` | Class `TestHalvingSearch` | MUST | T-1056 |  |
| F-1057 | `test_tuner.py` | Class `TestOptuna` | MUST | T-1057 |  |
| F-1058 | `test_tuner.py` | Class `TestBayesSearch` | MUST | T-1058 |  |
| F-1059 | `test_tuner.py` | Class `TestErrorHandling` | MUST | T-1059 |  |
| F-1060 | `test_tuner.py` | Class `TestCVResults` | MUST | T-1060 |  |
| F-1061 | `test_tuner.py` | `regression_data`() | MUST | T-1061 |  |
| F-1062 | `test_tuner.py` | `classification_data`() | MUST | T-1062 |  |
| F-1063 | `test_tuner_comprehensive.py` | Class `TestTunerConfig` | MUST | T-1063 |  |
| F-1064 | `test_tuner_comprehensive.py` | Class `TestConvertOptunaGrid` | MUST | T-1064 |  |
| F-1065 | `test_tuner_comprehensive.py` | Class `TestTuneGrid` | MUST | T-1065 |  |
| F-1066 | `test_tuner_comprehensive.py` | Class `TestTuneUnknown` | MUST | T-1066 |  |
| F-1067 | `test_tuner_comprehensive.py` | Class `TestTuneHalving` | MUST | T-1067 |  |
| F-1068 | `test_tuner_comprehensive.py` | Class `TestTuneOptuna` | MUST | T-1068 |  |
| F-1069 | `test_tuner_comprehensive.py` | Class `TestTuneBayes` | MUST | T-1069 |  |
| F-1070 | `test_tuner_extra.py` | Class `TestTunerConfig` | MUST | T-1070 |  |
| F-1071 | `test_tuner_extra.py` | Class `TestConvertOptunaGrid` | MUST | T-1071 |  |
| F-1072 | `test_tuner_extra.py` | Class `TestTuneGrid` | MUST | T-1072 |  |
| F-1073 | `test_tuner_extra.py` | Class `TestTuneRandom` | MUST | T-1073 |  |
| F-1074 | `test_tuner_extra.py` | Class `TestTuneHalving` | MUST | T-1074 |  |
| F-1075 | `test_tuner_extra.py` | Class `TestTuneOptuna` | MUST | T-1075 |  |
| F-1076 | `test_tuner_extra.py` | Class `TestTuneBayes` | MUST | T-1076 |  |
| F-1077 | `test_tuner_extra.py` | Class `TestTuneUnknown` | MUST | T-1077 |  |
| F-1078 | `test_tuner_pipeline.py` | Class `TestDetectPipelineStepName` | MUST | T-1078 |  |
| F-1079 | `test_tuner_pipeline.py` | Class `TestPrefixParamGrid` | MUST | T-1079 |  |
| F-1080 | `test_tuner_pipeline.py` | Class `TestStripPrefix` | MUST | T-1080 |  |
| F-1081 | `test_tuner_pipeline.py` | Class `TestTuneWithPipeline` | MUST | T-1081 |  |
| F-1082 | `test_type_detector_comprehensive.py` | Class `TestColumnInfo` | MUST | T-1082 |  |
| F-1083 | `test_type_detector_comprehensive.py` | Class `TestTypeDetector` | MUST | T-1083 |  |
| F-1084 | `test_type_detector_comprehensive.py` | Class `TestDetectionResult` | MUST | T-1084 |  |
| F-1085 | `test_type_detector_extra.py` | Class `TestColumnInfo` | MUST | T-1085 |  |
| F-1086 | `test_type_detector_extra.py` | Class `TestTypeDetector` | MUST | T-1086 |  |
| F-1087 | `test_type_detector_extra.py` | Class `TestDetectionResult` | MUST | T-1087 |  |
| F-1088 | `test_uma_adapter.py` | Class `TestUMAAdapterProperties` | MUST | T-1088 |  |
| F-1089 | `test_uma_adapter.py` | Class `TestUMAAdapterAvailability` | MUST | T-1089 |  |
| F-1090 | `test_uma_adapter.py` | Class `TestUMAAdapterDescriptors` | MUST | T-1090 |  |
| F-1091 | `test_uma_adapter.py` | Class `TestSmilesToAseAtoms` | MUST | T-1091 |  |
| F-1092 | `test_uma_adapter.py` | Class `TestUMAAdapterCompute` | MUST | T-1092 |  |
| F-1093 | `test_uma_adapter.py` | Class `TestUMAInitIntegration` | MUST | T-1093 |  |
| F-1094 | `test_utils_and_base.py` | Class `TestConfig` | MUST | T-1094 |  |
| F-1095 | `test_utils_and_base.py` | Class `TestOptionalImport` | MUST | T-1095 |  |
| F-1096 | `test_utils_and_base.py` | Class `TestDescriptorResult` | MUST | T-1096 |  |
| F-1097 | `test_utils_and_base.py` | Class `TestDescriptorMetadata` | MUST | T-1097 |  |
| F-1098 | `test_utils_and_base.py` | Class `TestBaseChemAdapter` | MUST | T-1098 |  |
| F-1099 | `test_utils_and_base.py` | Class `TestBenchmarkDatasets` | MUST | T-1099 |  |
| F-1100 | `test_utils_and_base.py` | Class `TestMolAITokenizer` | MUST | T-1100 |  |
| F-1101 | `test_utils_and_base.py` | Class `TestMolAIAdapter` | MUST | T-1101 |  |
| F-1102 | `verify_automl_smiles_fix.py` | `test_automl_smiles_only`() | MUST | T-1102 |  |