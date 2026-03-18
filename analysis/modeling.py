import os
import json
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocessing.preprocess_metadata import preprocess_metadata
from .preprocessing.preprocess_kinematics import preprocess_kinematics
from .preprocessing.preprocess_syllables import preprocess_syllables
from .utils.run_kfold_cv import run_kfold_cv
from .plot.plot_2D import plot_2D
from .plot.plot_1D import plot_1D
from .plot.plot_class_distribution import plot_class_distribution
from .utils.get_best_dims import get_best_dims

def modeling(embeddings, metadata, kinematics, syllable_labels, token_shapes, args, layer_window_maps=None, data_type="raw"):
    kinematics_keys = list(kinematics[list(kinematics.keys())[0]].keys()) if kinematics is not None else []
    metadata_keys = metadata.columns.tolist() if metadata is not None else []
    # Linear/logistic regression analysis to predict metadata variables from the embeddings
    # Per Layer
    meta_results = {}
    kinematics_results = {}
    output_dir = os.path.join(args.output_dir, data_type)
    plot_1D(embeddings, token_shapes, kinematics=kinematics, output_path=os.path.join(output_dir, "figures", "plot_1D.png"))    

    for layer_key, embeddings_dict in embeddings.items():
        fig_dir = os.path.join(output_dir, "figures", layer_key)
        os.makedirs(fig_dir, exist_ok=True)
        layer_meta_results = {}
        layer_kinematics_results = {}
        token_shape = token_shapes[int(layer_key.split("_")[-1])]
        layer_window_map = layer_window_maps[layer_key] if layer_window_maps is not None else None

        # Per Metadata variable (except run_id and trial_id)
        for meta_var in metadata_keys:
            if meta_var == "run_id" or meta_var == "trial_id" or meta_var == "animal_id": # skip animal_id for now TODO
                continue
            print(f"Running analysis for {layer_key} - {meta_var}")
            X, y, groups, X_plot, y_plot = preprocess_metadata(args, embeddings_dict, metadata, meta_var, token_shape, layer_window_map)
            if layer_key == "layer_0":  # only plot for the first layer 
                plot_class_distribution(y, meta_var, output_dir)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            dummy_model = DummyClassifier(strategy="most_frequent")
            run_kfold_cv(layer_meta_results, meta_var, model, dummy_model, X, y, groups=groups, is_classification=True)
            dims = get_best_dims(layer_meta_results[meta_var], n_plot_features=X.shape[1])
            plot_2D(X_plot, y_plot, is_discrete=True, dims=dims, title=f"{meta_var} - {layer_key}", output_path=os.path.join(fig_dir, f"{meta_var}_{layer_key}.png"))

            # get the dimensions with the highest absolute coefficient values and plot them


        
        if kinematics is not None:
            X, y = preprocess_kinematics(embeddings_dict, kinematics, token_shape, kinematics_keys, args, layer_window_map)
        # Kinematics prediction (e.g. speed, acceleration) with linear regression
        for kin_var in kinematics_keys[:6]:  # first 6 columns of kinematics are of main interest
            print(f"Running analysis for {layer_key} - {kin_var}")
            y_kin = y[:, kinematics_keys.index(kin_var)]  # shape (n_tokens,) - select the kinematic variable to predict
            # Similar analysis for kinematic variables
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression())
            ])
            dummy_model = DummyRegressor(strategy="mean")
            run_kfold_cv(layer_kinematics_results, kin_var, model, dummy_model, X, y_kin, is_classification=False)
            # get dimensions with highest absolute coefficient values and plot them
            dims = get_best_dims(layer_kinematics_results[kin_var], n_plot_features=X.shape[1])
            plot_2D(X, y_kin, is_discrete=False, dims=dims, title=f"{kin_var} - {layer_key}", output_path=os.path.join(fig_dir, f"{kin_var}_{layer_key}.png"))
        kinematics_results[layer_key] = layer_kinematics_results

        # Syllable prediction with logistic regression
        if syllable_labels is not None:
            print(f"Running analysis for {layer_key} - syllable labels")
            X, y, groups = preprocess_syllables(embeddings_dict, syllable_labels, metadata, token_shape, args, layer_window_map)
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000))
            ])
            dummy_model = DummyClassifier(strategy="most_frequent")
            run_kfold_cv(layer_meta_results, "syllable", model, dummy_model, X, y, groups=groups, is_classification=True)
            # get dimensions with highest absolute coefficient values and plot them
            dims = get_best_dims(layer_meta_results["syllable"], n_plot_features=X.shape[1])
            plot_2D(X, y, is_discrete=True, dims=dims, title=f"syllable - {layer_key}", output_path=os.path.join(fig_dir, f"syllable_{layer_key}.png"))

        meta_results[layer_key] = layer_meta_results



    # Save results 
    if metadata_keys is not []:
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "embedding_analysis_meta_results.json")
        with open(results_file, "w") as f:
            json.dump(meta_results, f, indent=4)
        print(f"Saved metadata embedding analysis results to {results_file}")
    if kinematics is not None:
        kin_results_file = os.path.join(output_dir, "embedding_analysis_kinematics_results.json")
        with open(kin_results_file, "w") as f:
            json.dump(kinematics_results, f, indent=4)
        print(f"Saved kinematics embedding analysis results to {kin_results_file}")